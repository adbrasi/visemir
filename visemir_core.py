"""Lógica do visemir: detecção da boca, grid de visemas, edição (fal/local), máscaras e tracking planar.

Todas as imagens aqui são numpy uint8 RGB (H, W, 3). Usado pelo CLI (cli.py) e pelos nodes do ComfyUI (nodes.py).
"""
import base64, io, json, os, sys, time, urllib.error, urllib.request

import cv2
import numpy as np
from PIL import Image

VISEMES = [  # (nome, descrição exata para o modelo de imagem)
    ('MBP', 'lips closed together'),
    ('A',   'mouth wide open, jaw lowered, tongue visible'),
    ('E',   'mouth open wide horizontally, upper and lower teeth visible'),
    ('O',   'mouth open in a round O shape'),
    ('U',   'small round pursed lips'),
    ('FV',  'upper teeth resting on the lower lip'),
]
NAMES = [n for n, _ in VISEMES]
REST = 'REST'  # boca original
DEFAULT_SEQ = 'REST,MBP,A,O,MBP,E,U,A,FV,O,MBP,REST'
DETECT_MODES = ['auto', 'mouth', 'nose-chin', 'face']
FLUX_URL = 'https://fal.run/blackforestlabs/flux-3/edit-image'
MOONDREAM_URL = 'https://api.moondream.ai/v1/detect'
UA = 'visemir/1.0'
CELL, LINE = 512, 16  # grid = 1568x1040: múltiplos de 16, direto para latente
GRID_W, GRID_H = CELL * 3 + LINE * 2, CELL * 2 + LINE
CC_LOW = 0.6  # abaixo disso o tracking é considerado perdido


class VisemirError(Exception):
    pass


class NotFound(VisemirError):
    """Nada detectado neste frame (dá para tentar outro frame)."""


def log(*a):
    print('[visemir]', *a, file=sys.stderr, flush=True)


def get_key(explicit, *env_names):
    if explicit and explicit.strip():
        return explicit.strip()
    for n in env_names:
        if os.environ.get(n):
            return os.environ[n].strip()
    raise VisemirError(f'chave não encontrada: passe no node/argumento ou defina a variável de ambiente {env_names[0]}')


def moondream_key(k=None): return get_key(k, 'MOONDREAM_API_KEY')
def fal_key(k=None): return get_key(k, 'FAL_API_KEY', 'FAL_KEY')


# ---------------------------------------------------------------- HTTP

def http_json(url, body, headers, timeout=600, retries=3, what='API'):
    data = json.dumps(body).encode()
    h = {'Content-Type': 'application/json', 'User-Agent': UA, **headers}
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h), timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors='replace')[:800]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                log(f'{what}: HTTP {e.code}, tentando de novo...'); time.sleep(2 ** (attempt + 1)); continue
            if 'content_policy' in msg:
                raise VisemirError(f'{what} recusou a imagem/prompt pelo filtro de conteúdo.')
            if e.code in (401, 403):
                raise VisemirError(f'{what}: acesso negado (HTTP {e.code}), verifique a chave. {msg[:200]}')
            raise VisemirError(f'{what}: HTTP {e.code}: {msg}')
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries:
                log(f'{what}: erro de rede ({e}), tentando de novo...'); time.sleep(2 ** (attempt + 1)); continue
            raise VisemirError(f'{what}: falha de rede: {e}')


def http_get(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 3:
                raise VisemirError(f'falha ao baixar {url}: {e}')
            time.sleep(2 ** (attempt + 1))


def data_uri(img, fmt='PNG'):
    b = io.BytesIO(); Image.fromarray(img).save(b, fmt)
    return f'data:image/{fmt.lower()};base64,' + base64.b64encode(b.getvalue()).decode()


def decode_image(raw):
    im = Image.open(io.BytesIO(raw))
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA'); bg = Image.new('RGB', im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3]); im = bg
    return np.asarray(im.convert('RGB')).copy()


# ---------------------------------------------------------------- detecção

def moondream(img, obj, key):
    """Caixas (x0,y0,x1,y1) em pixels da imagem."""
    H, W = img.shape[:2]
    small = Image.fromarray(img); small.thumbnail((1024, 1024), Image.LANCZOS)  # coordenadas voltam normalizadas
    r = http_json(MOONDREAM_URL, {'image_url': data_uri(np.asarray(small), 'JPEG'), 'object': obj},
                  {'X-Moondream-Auth': key}, timeout=120, what='Moondream')
    return [(o['x_min'] * W, o['y_min'] * H, o['x_max'] * W, o['y_max'] * H) for o in r.get('objects', [])]


def _area(b): return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
def _center(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
def _inside(p, b): return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def detect(img, key=None, mode='auto', face_frac=0.9):
    """Acha a boca: boca -> média nariz/queixo -> rosto inteiro. Devolve também o recorte quadrado (x0, y0, lado)."""
    if mode not in DETECT_MODES:
        raise VisemirError(f'modo inválido {mode!r}; use {DETECT_MODES}')
    key = moondream_key(key)
    H, W = img.shape[:2]
    if float(img.std()) < 4:  # frame preto/branco/fade: nem pergunta (o modelo alucina caixas aqui)
        raise NotFound('frame praticamente uniforme')
    full = W * H
    # descarta caixas absurdas (ex.: caixa do tamanho da imagem inteira)
    faces = [b for b in moondream(img, 'face', key) if not (b[2] - b[0] > 0.97 * W and b[3] - b[1] > 0.97 * H)]
    face = max(faces, key=_area) if faces else None

    def pick(boxes, max_frac=0.15):
        boxes = [b for b in boxes if 0 < _area(b) <= max_frac * full]
        if face:  # com rosto, só vale o que está dentro dele e menor que ele
            boxes = [b for b in boxes if _inside(_center(b), face) and _area(b) <= 0.5 * _area(face)]
        return max(boxes, key=_area) if boxes else None

    mouth = method = None
    if mode in ('auto', 'mouth'):
        mouth = pick(moondream(img, 'mouth', key)); method = 'mouth' if mouth else None
    if mouth is None and mode in ('auto', 'nose-chin'):
        nose, chin = pick(moondream(img, 'nose', key)), pick(moondream(img, 'chin', key), 0.4)
        if nose and chin:
            # média entre a base do nariz e o centro do queixo (medido: cai no centro da boca)
            cx = (_center(nose)[0] + _center(chin)[0]) / 2
            cy = (nose[3] + _center(chin)[1]) / 2
            d = max(_center(chin)[1] - _center(nose)[1], 1)
            mw = (face[2] - face[0]) * 0.1 if face else d * 0.27
            mh = (face[3] - face[1]) * 0.045 if face else d * 0.13
            mouth = (cx - mw / 2, cy - mh / 2, cx + mw / 2, cy + mh / 2); method = 'nose-chin'
    if mouth is None and mode in ('auto', 'face') and face:
        fw, fh = face[2] - face[0], face[3] - face[1]
        cx, cy = (face[0] + face[2]) / 2, face[1] + 0.78 * fh  # boca de anime fica a ~78% da altura do rosto
        mouth = (cx - fw * 0.05, cy - fh * 0.0225, cx + fw * 0.05, cy + fh * 0.0225); method = 'face'
    if mouth is None:
        raise NotFound('não encontrei boca, nariz/queixo nem rosto' if mode == 'auto' else f'o modo "{mode}" não encontrou nada')

    return finish_detection(mouth, face, method, W, H, face_frac)


def manual_detection(img, mouth, face=None, face_frac=0.9):
    """Detecção informada à mão (caixas x0,y0,x1,y1 em pixels): dispensa o Moondream."""
    H, W = img.shape[:2]
    for name, b in (('boca', mouth), ('rosto', face)):
        if b is not None and not (0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H):
            raise VisemirError(f'caixa da {name} {list(b)} fora da imagem {W}x{H} ou com x0>=x1 / y0>=y1')
    return finish_detection(tuple(mouth), tuple(face) if face else None, 'manual', W, H, face_frac)


def finish_detection(mouth, face, method, W, H, face_frac):
    """Recorte quadrado (x0, y0, lado) em volta da boca, do tamanho do rosto."""
    cx, cy = _center(mouth)
    side = (face[3] - face[1]) * face_frac if face else (mouth[2] - mouth[0]) * 10 * face_frac
    side = int(round(min(max(side, (mouth[2] - mouth[0]) * 3, 64), W, H)))
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side * 0.45, 0), H - side)))
    r = lambda b: [round(float(v), 1) for v in b] if b else None
    return {'method': method, 'mouth': r(mouth), 'face': r(face), 'crop': [x0, y0, side], 'image_size': [W, H]}


def find_reference(get_frame, n_frames, start=0, step=12, max_attempts=5, key=None, mode='auto', face_frac=0.9):
    """Procura a boca a partir de `start`; se não achar, pula `step` frames, até `max_attempts` tentativas.
    get_frame(i) -> RGB uint8. Devolve (índice, frame, detecção)."""
    if n_frames < 1:
        raise VisemirError('o vídeo não tem frames')
    if not 0 <= start < n_frames:
        raise VisemirError(f'frame inicial {start} fora do vídeo (0..{n_frames - 1})')
    tried = []
    for a in range(max(1, max_attempts)):
        i = start + a * max(1, step)
        if i >= n_frames:
            break
        fr = get_frame(i)
        try:
            det = detect(fr, key, mode, face_frac)
            if a:
                log(f'boca encontrada no frame {i} (frames sem boca: {tried})')
            return i, fr, det
        except NotFound:
            tried.append(i); log(f'frame {i}: nada encontrado, tentando frame {i + step}...')
    raise NotFound(f'não encontrei boca/rosto nos frames {tried}')


def draw_detection(img, det):
    out = img.copy(); w = max(2, img.shape[1] // 300)
    b = lambda r, c, t: cv2.rectangle(out, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), c, t)
    if det['face']: b(det['face'], (255, 255, 0), w)
    b(det['mouth'], (0, 255, 0), w)
    x0, y0, s = det['crop']; b((x0, y0, x0 + s, y0 + s), (255, 0, 0), w + 1)
    return out


# ---------------------------------------------------------------- grid e edição

def crop_of(img, det):
    x0, y0, s = det['crop']
    return img[y0:y0 + s, x0:x0 + s]


def build_grid(img, det):
    """Recorte repetido em 6 quadrados (3x2) separados por linhas brancas. Saída GRID_W x GRID_H."""
    tile = cv2.resize(crop_of(img, det), (CELL, CELL), interpolation=cv2.INTER_LANCZOS4)
    g = np.full((GRID_H, GRID_W, 3), 255, np.uint8)
    for i in range(6):
        x, y = (i % 3) * (CELL + LINE), (i // 3) * (CELL + LINE)
        g[y:y + CELL, x:x + CELL] = tile
    return g


def build_prompt(style='anime'):
    d = [desc for _, desc in VISEMES]
    return (f'3x2 grid of the same {style} face crop, tiles separated by white lines. Keep the white lines. '
            f'Change only the mouth in each tile, everything else identical.\n'
            f'Top row: tile 1 {d[0]}; tile 2 {d[1]}; tile 3 {d[2]}.\n'
            f'Bottom row: tile 1 {d[3]}; tile 2 {d[4]}; tile 3 {d[5]}.\n'
            f'Same line art, same colors, same nose, chin, neck and hair, same position in every tile.')


def fal_edit(grid, prompt, key=None):
    r = http_json(FLUX_URL, {'prompt': prompt, 'image_urls': [data_uri(grid)], 'resolution': '1.5k',
                             'aspect_ratio': '3:2', 'output_format': 'png'},
                  {'Authorization': 'Key ' + fal_key(key)}, timeout=600, what='FLUX (fal)')
    try:
        url = r['images'][0]['url']
    except (KeyError, IndexError, TypeError):
        raise VisemirError(f'resposta inesperada do FLUX: {json.dumps(r)[:300]}')
    return decode_image(http_get(url))


def split_grid(g):
    """Corta os 6 quadrados achando as linhas brancas (qualquer resolução); sem linhas, divide igualmente."""
    G = g.astype(np.float32).min(2)

    def spans(prof, n):
        L = len(prof); white = prof > 235; win = max(int(L * 0.04), 4); cuts = []
        for k in range(1, n):
            c = int(L * k / n); lo = max(c - win, 0)
            w = np.where(white[lo:c + win])[0] + lo
            cuts.append((int(w.min()), int(w.max()) + 1) if len(w) else (c, c))
        e = [0] + [x for c in cuts for x in c] + [L]
        return [(e[2 * j], e[2 * j + 1]) for j in range(n)]

    cols, rows = spans(G.mean(0), 3), spans(G.mean(1), 2)
    m = 2  # descarta a borda anti-aliased da linha
    tiles = []
    for i in range(6):
        (x0, x1), (y0, y1) = cols[i % 3], rows[i // 3]
        tiles.append(g[y0 + (m if i // 3 else 0):y1 - (m if i // 3 == 0 else 0),
                       x0 + (m if i % 3 else 0):x1 - (m if i % 3 < 2 else 0)])
    return tiles


def _ellipse(S, cx, cy, rx, ry):
    yy, xx = np.mgrid[:S, :S]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1


def make_visemes(ref, grid_out, det):
    """Para cada quadrado: acha a região que mais mudou perto da boca, mantém só a ilha principal e suaviza.
    Devolve lista de 6 dicts: image (frame inteiro com a boca), mask (float 0..1, tamanho do frame), sprite (RGBA do recorte)."""
    x0, y0, S = det['crop']
    refc = crop_of(ref, det); refa = refc.astype(np.float32)
    mx0, my0, mx1, my1 = det['mouth']; MX, MY = (mx0 + mx1) / 2 - x0, (my0 + my1) / 2 - y0
    k = S / 384  # parâmetros calibrados num recorte de 384px
    roi = _ellipse(S, MX, MY, 0.31 * S, 0.26 * S)
    mouth0 = _ellipse(S, MX, MY, (mx1 - mx0) / 2 * 1.15 + 2, (my1 - my0) / 2 * 1.15 + 2)
    r = max(1, round(10 * k)); ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    out = []
    for t in split_grid(grid_out):
        t = cv2.resize(t, (S, S), interpolation=cv2.INTER_LANCZOS4)
        d = cv2.GaussianBlur(np.abs(t.astype(np.float32) - refa).max(2), (0, 0), 2 * k)
        d[~roi] = 0
        m = (d > 40).astype(np.uint8)
        n, lab = cv2.connectedComponents(m, connectivity=8)
        if n > 1:  # região com a maior mudança acumulada
            s = np.bincount(lab.ravel(), weights=d.ravel(), minlength=n); s[0] = 0
            m = (lab == int(s.argmax())).astype(np.uint8)
        m = cv2.dilate(m | mouth0.astype(np.uint8), ker)
        pm = np.pad(m, 1); cv2.floodFill(pm, None, (0, 0), 2); m = (pm != 2)[1:-1, 1:-1].astype(np.float32)  # tapa buracos
        a = cv2.GaussianBlur(m * 255, (0, 0), 8 * k)
        n, lab = cv2.connectedComponents((a >= 1).astype(np.uint8), connectivity=8)
        if n > 2:  # limpeza: tudo que não está conectado à máscara principal sai
            a[lab != lab[np.unravel_index(a.argmax(), a.shape)]] = 0
        a = np.clip(a / 255, 0, 1)
        full = ref.copy()
        full[y0:y0 + S, x0:x0 + S] = (t * a[..., None] + refa * (1 - a[..., None])).round().astype(np.uint8)
        fm = np.zeros(ref.shape[:2], np.float32); fm[y0:y0 + S, x0:x0 + S] = a
        sprite = np.dstack([t, (a * 255).round().astype(np.uint8)])
        out.append({'image': full, 'mask': fm, 'sprite': sprite})
    return out


def preview_sheet(visemes, det):
    x0, y0, S = det['crop']
    sheet = np.zeros((S * 2, S * 3, 3), np.uint8)
    for i, v in enumerate(visemes):
        sheet[(i // 3) * S:(i // 3 + 1) * S, (i % 3) * S:(i % 3 + 1) * S] = crop_of(v['image'], det)
    return sheet


# ---------------------------------------------------------------- tracking

def parse_seq(s):
    seq = [x.strip().upper() for x in str(s).split(',') if x.strip()]
    bad = [x for x in seq if x not in NAMES and x != REST]
    if bad or not seq:
        raise VisemirError(f'sequência inválida {bad or s!r}. Use: {", ".join(NAMES + [REST])}')
    return seq


def track_roi(det, masks, shape):
    """Região que o tracker segura: o recorte (nariz->pescoço) menos tudo que as bocas tocam."""
    x0, y0, S = det['crop']
    roi = np.zeros(shape[:2], np.uint8); roi[y0:y0 + S, x0:x0 + S] = 255
    anym = (np.max(masks, 0) > 0.01).astype(np.uint8)
    r = max(4, S // 40)
    roi[cv2.dilate(anym, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0] = 0
    return roi


def _ecc(tpl, img, tpl_mask, H, iters, eps, gauss):
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    if hasattr(cv2, 'findTransformECCWithMask'):
        return cv2.findTransformECCWithMask(tpl, img, tpl_mask, np.full_like(img, 255), H.astype(np.float32),
                                            cv2.MOTION_HOMOGRAPHY, crit, gauss)
    # OpenCV antigo: a máscara é no frame; projeta a máscara do template com a estimativa atual
    m = cv2.warpPerspective(tpl_mask, H.astype(np.float64), (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
    return cv2.findTransformECC(tpl, img, H.astype(np.float32), cv2.MOTION_HOMOGRAPHY, crit, m, gauss)


class Tracker:
    """Tracker planar (estilo Mocha): KLT prevê, ECC contra a referência refina no subpixel sem drift,
    SIFT relocaliza na inicialização e quando perde (cortes, saltos)."""

    def __init__(self, ref_gray, roi):
        self.ref = ref_gray; self.roi = roi
        self.pts = cv2.goodFeaturesToTrack(ref_gray, 400, 0.01, 6, mask=roi)
        if self.pts is None or len(self.pts) < 8:
            raise VisemirError('a região do rosto tem poucos detalhes para rastrear')
        self.sift = cv2.SIFT_create(2000)
        self.kp, self.des = self.sift.detectAndCompute(ref_gray, roi)
        ys, xs = np.where(roi > 0)  # ECC só no retângulo da região (bem mais rápido)
        bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        self.T = np.array([[1, 0, bx0], [0, 1, by0], [0, 0, 1]], np.float64); self.Ti = np.linalg.inv(self.T)
        self.tpl = ref_gray[by0:by1, bx0:bx1].copy(); self.tpl_roi = roi[by0:by1, bx0:bx1].copy()
        self.s = min(1.0, 256 / max(self.tpl.shape))  # passada grossa do ECC
        self.tpl_s = cv2.resize(self.tpl, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)
        self.roi_s = cv2.resize(self.tpl_roi, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_NEAREST)
        self.reset()

    def reset(self, H=None, prev=None):
        """H = estimativa ref->frame para o próximo frame; prev = frame anterior (cinza) para o optical flow."""
        self.H = None if H is None else np.asarray(H, np.float64)
        self.prev = prev if H is not None else None
        self.prev_pts = cv2.perspectiveTransform(self.pts, self.H).astype(np.float32) if H is not None else None

    def relocalize(self, g):
        kp, des = self.sift.detectAndCompute(g, None)
        if des is None or self.des is None or len(kp) < 8:
            return None
        good = [a for a, b in (x for x in cv2.BFMatcher().knnMatch(self.des, des, k=2) if len(x) == 2) if a.distance < 0.75 * b.distance]
        if len(good) < 10:
            return None
        src = cv2.KeyPoint_convert([self.kp[x.queryIdx] for x in good]).reshape(-1, 1, 2)
        dst = cv2.KeyPoint_convert([kp[x.trainIdx] for x in good]).reshape(-1, 1, 2)
        return cv2.findHomography(src, dst, cv2.RANSAC, 4.0)[0]

    def ecc(self, g, H):
        S = np.diag([self.s, self.s, 1.0]); Si = np.linalg.inv(S)
        try:
            Hc = H @ self.T  # template recortado -> frame
            gs = cv2.resize(g, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)
            cc, Hs = _ecc(self.tpl_s, gs, self.roi_s, S @ Hc @ Si, 50, 1e-5, 3)
            Hc = Si @ Hs.astype(np.float64) @ S
            if self.s < 1:
                cc, Hf = _ecc(self.tpl, g, self.tpl_roi, Hc, 15, 1e-6, 5)
                Hc = Hf.astype(np.float64)
            return float(cc), Hc @ self.Ti
        except cv2.error:
            return 0.0, H

    def update(self, g):
        H = self.H
        if H is not None and self.prev is not None:  # previsão por optical flow
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, self.prev_pts, None, winSize=(21, 21), maxLevel=4)
            st = st.ravel() == 1
            if st.sum() >= 8:
                Hk = cv2.findHomography(self.pts[st], nxt[st], cv2.RANSAC, 3.0)[0]
                if Hk is not None: H = Hk
        if H is None:
            H = self.relocalize(g)
            if H is None:
                h, w = g.shape; rh, rw = self.ref.shape
                H = np.diag([w / rw, h / rh, 1.0])
        cc, H = self.ecc(g, H)
        if cc < CC_LOW:  # perdeu: tenta relocalizar do zero
            Hr = self.relocalize(g)
            if Hr is not None:
                cc2, Hr = self.ecc(g, Hr)
                if cc2 > cc: cc, H = cc2, Hr
        self.reset(H, g)
        return H, cc


def gray(rgb): return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def make_tracker(ref, visemes, det):
    return Tracker(gray(ref), track_roi(det, [v['mask'] for v in visemes], ref.shape))


def init_H(frame, ref):
    """Identidade se o frame é a própria referência (mesmo vídeo); senão None (SIFT acha)."""
    if frame.shape == ref.shape and np.abs(frame.astype(np.int16) - ref).mean() < 3:
        return np.eye(3)
    return None


def composite(frame, viseme, H):
    h, w = frame.shape[:2]
    wp = cv2.warpPerspective(viseme['image'], H, (w, h), flags=cv2.INTER_LINEAR)
    wm = cv2.warpPerspective(viseme['mask'], H, (w, h), flags=cv2.INTER_LINEAR)[..., None]
    return (wp * wm + frame * (1 - wm)).round().astype(np.uint8)


def draw_track(frame, det, H, cc, label):
    x0, y0, S = det['crop']
    box = cv2.perspectiveTransform(np.float32([[x0, y0], [x0 + S, y0], [x0 + S, y0 + S], [x0, y0 + S]])[None], H)[0]
    out = frame.copy()
    col = (255, 255, 0) if cc >= CC_LOW else (255, 0, 0)
    cv2.polylines(out, [box.astype(np.int32)], True, col, 2)
    cv2.putText(out, f'{label}  cc={cc:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    return out


def viseme_at(seq, hold, k):
    return seq[(k // max(1, hold)) % len(seq)]


def track_all(get_frame, n, ref, ref_index, tracker, progress=None):
    """Homografias ref->frame para todos os frames: anda para frente a partir do frame de referência e depois para trás."""
    Hs, ccs = [None] * n, [0.0] * n
    ref_index = min(max(ref_index or 0, 0), n - 1)
    fr = get_frame(ref_index); tracker.reset(init_H(fr, ref))
    for i in range(ref_index, n):
        g = gray(get_frame(i) if i != ref_index else fr)
        Hs[i], ccs[i] = tracker.update(g)
        if i == ref_index: g_ref = g
        if progress: progress()
    if ref_index > 0:
        tracker.reset(Hs[ref_index], g_ref)
        for i in range(ref_index - 1, -1, -1):
            Hs[i], ccs[i] = tracker.update(gray(get_frame(i)))
            if progress: progress()
    return Hs, ccs


# ---------------------------------------------------------------- exportação para engines (Phaser / Godot)

def trimmed_sprites(ref, visemes):
    """Cada boca recortada justa (RGBA) + posição exata dentro da imagem base."""
    out = []
    for name, v in zip(NAMES, visemes):
        a = (v['mask'] * 255).round().astype(np.uint8)
        ys, xs = np.where(a > 0)
        if len(xs) == 0:
            raise VisemirError(f'a boca {name} ficou com máscara vazia')
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        rgba = np.dstack([v['image'][y0:y1, x0:x1], a[y0:y1, x0:x1]])
        out.append({'name': name, 'rgba': rgba, 'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0})
    return out


def build_atlas(sprites, pad=2):
    """Empacota as bocas numa linha (são poucas e pequenas). Devolve (atlas RGBA, posições no atlas)."""
    W = sum(s['w'] + pad for s in sprites) + pad; H = max(s['h'] for s in sprites) + pad * 2
    atlas = np.zeros((H, W, 4), np.uint8); pos = []; x = pad
    for s in sprites:
        atlas[pad:pad + s['h'], x:x + s['w']] = s['rgba']; pos.append((x, pad)); x += s['w'] + pad
    return atlas, pos


def export_files(ref, visemes, det, atlas_name='visemes_atlas', godot_res_dir='res://'):
    """Monta todos os arquivos de exportação em memória: {caminho_relativo: bytes}.
    - base.png: personagem com a boca original (REST)
    - <atlas>.png + <atlas>.json: atlas Phaser (JSON Hash, trimmed: a posição na base já vem no frame)
    - sprites/<NOME>.png: cada boca recortada justa
    - godot/<NOME>.tres: AtlasTexture (region + margin = mesma posição na base)
    - visemes.json: coordenadas exatas de tudo"""
    H, W = ref.shape[:2]
    sp = trimmed_sprites(ref, visemes)
    atlas, pos = build_atlas(sp)
    png = lambda arr: (lambda b: (Image.fromarray(arr).save(b, 'PNG'), b.getvalue())[1])(io.BytesIO())
    files = {'base.png': png(ref), f'{atlas_name}.png': png(atlas)}
    frames, coords = {}, {}
    for s, (ax, ay) in zip(sp, pos):
        n = s['name']
        frames[n] = {'frame': {'x': ax, 'y': ay, 'w': s['w'], 'h': s['h']}, 'rotated': False, 'trimmed': True,
                     'spriteSourceSize': {'x': s['x'], 'y': s['y'], 'w': s['w'], 'h': s['h']},
                     'sourceSize': {'w': W, 'h': H}}
        coords[n] = {'x': s['x'], 'y': s['y'], 'w': s['w'], 'h': s['h'],
                     'center': [s['x'] + s['w'] / 2, s['y'] + s['h'] / 2], 'atlas': {'x': ax, 'y': ay},
                     'sprite': f'sprites/{n}.png'}
        files[f'sprites/{n}.png'] = png(s['rgba'])
        rd = godot_res_dir if godot_res_dir.endswith('/') else godot_res_dir + '/'
        files[f'godot/{n}.tres'] = (
            '[gd_resource type="AtlasTexture" load_steps=2 format=3]\n\n'
            f'[ext_resource type="Texture2D" path="{rd}{atlas_name}.png" id="1"]\n\n'
            '[resource]\natlas = ExtResource("1")\n'
            f'region = Rect2({ax}, {ay}, {s["w"]}, {s["h"]})\n'
            f'margin = Rect2({s["x"]}, {s["y"]}, {W - s["w"]}, {H - s["h"]})\n').encode()
    # `visemir` é uma chave extra no topo: o Phaser a copia para texture.customData ao carregar o atlas,
    # então a engine lê a caixa da boca (e o tracking, ver embed_track) do mesmo arquivo do atlas.
    files[f'{atlas_name}.json'] = json.dumps({'frames': frames, 'meta': {
        'app': 'visemir', 'version': '1.0', 'image': f'{atlas_name}.png', 'format': 'RGBA8888',
        'size': {'w': int(atlas.shape[1]), 'h': int(atlas.shape[0])}, 'scale': '1'},
        'visemir': {'version': 1, 'rest': REST, 'mouth': det['mouth'], 'size': [W, H]}}, indent=2).encode()
    files['visemes.json'] = json.dumps({
        'base': {'image': 'base.png', 'w': W, 'h': H}, 'order': NAMES, 'rest': REST,
        'mouth': det['mouth'], 'crop': det['crop'], 'visemes': coords,
        'note': 'x,y = canto superior esquerdo da boca na imagem base, em pixels. Desenhe a boca em cima da base nessa posição.'},
        indent=2).encode()
    return files, atlas


def embed_track(atlas_json, fps, size, Hs, ccs):
    """Grava o tracking de um vídeo dentro do JSON do atlas (chave visemir.track), para a engine colar a boca
    no vídeo em tempo real: homografias referência->frame (3x3 em linha, pixels do vídeo) + confiança por frame."""
    data = json.loads(atlas_json)
    vis = data.setdefault('visemir', {'version': 1, 'rest': REST})
    g = lambda v: float(f'{float(v):.6g}')
    vis['track'] = {'fps': g(fps), 'size': [int(size[0]), int(size[1])], 'low': CC_LOW,
                    'homographies': [[g(v) for v in np.asarray(h, np.float64).ravel()] for h in Hs],
                    'confidence': [round(float(c), 3) for c in ccs]}
    return json.dumps(data, indent=1).encode()


def write_files(files, out_dir):
    from pathlib import Path
    out = Path(out_dir)
    for rel, data in files.items():
        p = out / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    return out
