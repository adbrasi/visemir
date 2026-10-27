#!/usr/bin/env python3
"""visemir — 6 visemas (bocas) de um personagem: detecção, geração, máscaras, tracking em vídeo e exportação para engines.

Chaves pelas variáveis de ambiente MOONDREAM_API_KEY e FAL_API_KEY.

Comandos:
  doctor     verifica dependências, ffmpeg e chaves
  detect     acha a boca (boca -> nariz/queixo -> rosto) e a área de recorte
  prepare    só monta o grid + prompt + tamanho (para rodar num modelo de edição local)
  generate   prepare + FLUX 3 (fal) ou --edited GRID + máscaras + visemas + exportação Phaser/Godot
  track      cola os visemas num vídeo com tracking
  demo       cria um vídeo de teste da imagem se mexendo em zigzag
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

VIDEO_EXT = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.gif'}


def die(msg):
    print(f'erro: {msg}', file=sys.stderr); sys.exit(1)


try:
    import numpy as np, cv2
    from PIL import Image
    import visemir_core as core
except ImportError as e:
    die(f'dependência faltando ({e.name}). Rode: pip install -r {Path(__file__).resolve().parent / "requirements.txt"}')

log = core.log


class Source:
    """Imagem ou vídeo, com acesso a frames por índice (sequencial é rápido; pular faz seek)."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise core.VisemirError(f'arquivo não encontrado: {path}')
        self.is_video = self.path.suffix.lower() in VIDEO_EXT
        if self.is_video:
            self.cap = cv2.VideoCapture(str(self.path))
            if not self.cap.isOpened():
                raise core.VisemirError(f'não consegui abrir o vídeo {path}')
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.pos = 0
            if self.n <= 0:  # alguns containers não informam: conta
                while self.cap.grab(): self.n += 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            try:
                self.img = core.decode_image(self.path.read_bytes())
            except Exception as e:
                raise core.VisemirError(f'não consegui abrir a imagem {path}: {e}')
            self.n, self.fps = 1, 30

    def get(self, i):
        if not self.is_video:
            return self.img
        if i != self.pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = self.cap.read()
        if not ok:
            raise core.VisemirError(f'não consegui ler o frame {i} de {self.path}')
        self.pos = i + 1
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def save(path, rgb):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def load(path):
    return core.decode_image(Path(path).read_bytes())


def save_gif(frames, path, ms, max_side=512):
    """GIF com paleta única para todos os frames (senão a imagem inteira 'treme')."""
    fr = []
    for f in frames:
        im = Image.fromarray(f); im.thumbnail((max_side, max_side), Image.LANCZOS); fr.append(im)
    W, H = fr[0].size; strip = Image.new('RGB', (W, H * len(fr)))
    for i, f in enumerate(fr): strip.paste(f, (0, i * H))
    pal = strip.quantize(256, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in fr]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0, optimize=False)


def open_writer(path, W, H, fps, audio_src=None):
    if shutil.which('ffmpeg'):
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-']
        if audio_src:
            cmd += ['-i', str(audio_src), '-map', '0:v', '-map', '1:a?', '-c:a', 'aac', '-shortest']
        cmd += ['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', str(path)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return (lambda fr: p.stdin.write(np.ascontiguousarray(fr).tobytes())), (lambda: (p.stdin.close(), p.wait()))
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    return (lambda fr: w.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))), w.release


def find_ref(a, src):
    return core.find_reference(src.get, src.n, a.frame, a.retry_step, a.max_attempts, None, a.mode, a.face_frac)


# ---------------------------------------------------------------- comandos

def cmd_doctor(a):
    import os
    ok = True
    for mod, pkg in (('numpy', 'numpy'), ('PIL', 'pillow'), ('cv2', 'opencv-python')):
        try:
            __import__(mod); print(f'  ok    {pkg}')
        except ImportError:
            print(f'  FALTA {pkg}'); ok = False
    print(f'  ok    opencv {cv2.__version__} (ECC com máscara: {"sim" if hasattr(cv2, "findTransformECCWithMask") else "fallback"})')
    print(f'  {"ok   " if shutil.which("ffmpeg") else "aviso"} ffmpeg{"" if shutil.which("ffmpeg") else " (sem ffmpeg: vídeo sai em mp4v e sem áudio)"}')
    for k in ('MOONDREAM_API_KEY', 'FAL_API_KEY'):
        has = bool(os.environ.get(k) or (k == 'FAL_API_KEY' and os.environ.get('FAL_KEY')))
        ok &= has; print(f'  {"ok   " if has else "FALTA"} {k}')
    print('tudo certo.' if ok else 'corrija os itens acima.')
    return 0 if ok else 1


def cmd_detect(a):
    src = Source(a.input)
    idx, fr, det = find_ref(a, src)
    det = {'frame': idx, **det}
    if a.debug:
        save(a.debug, core.draw_detection(fr, det)); log(f'debug salvo em {a.debug}')
    print(json.dumps(det, indent=2))


def write_prepare(out, src_path, idx, ref, det, style):
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'ref.png', ref); save(out / 'detection.png', core.draw_detection(ref, det))
    grid = core.build_grid(ref, det); save(out / 'grid_in.png', grid)
    prompt = core.build_prompt(style); (out / 'prompt.txt').write_text(prompt + '\n')
    meta = {'source': str(src_path), 'ref_frame': idx, 'detection': det, 'order': core.NAMES,
            'grid': {'image': 'grid_in.png', 'width': grid.shape[1], 'height': grid.shape[0]}, 'prompt': prompt}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    return grid, prompt, meta


def cmd_prepare(a):
    src = Source(a.input); out = Path(a.output)
    idx, ref, det = find_ref(a, src)
    log(f'frame {idx}: método {det["method"]}, boca {det["mouth"]}, recorte {det["crop"]}')
    grid, prompt, _ = write_prepare(out, a.input, idx, ref, det, a.style)
    log(f'pronto: {out / "grid_in.png"} ({grid.shape[1]}x{grid.shape[0]}) + prompt.txt')
    log(f'edite no seu modelo e depois: cli.py generate {a.input} -o {out} --edited GRID_EDITADO.png')


def cmd_generate(a):
    src = Source(a.input); out = Path(a.output)
    meta_p = out / 'meta.json'
    if a.edited and meta_p.is_file() and (out / 'ref.png').is_file():
        meta = json.loads(meta_p.read_text()); det, idx = meta['detection'], meta['ref_frame']
        ref = load(out / 'ref.png'); prompt = meta['prompt']
        log(f'usando a detecção salva em {meta_p} (frame {idx})')
    else:
        log('1/4 detectando boca...')
        idx, ref, det = find_ref(a, src)
        log(f'    frame {idx}: método {det["method"]}, boca {det["mouth"]}, recorte {det["crop"]}')
        grid, prompt, meta = write_prepare(out, a.input, idx, ref, det, a.style)
    if a.edited:
        p = Path(a.edited)
        if not p.is_file():
            raise core.VisemirError(f'grid editado não encontrado: {p}')
        edited = load(p)
        if p.resolve() != (out / 'grid_out.png').resolve():
            save(out / 'grid_out.png', edited)
    else:
        log('2/4 gerando bocas no FLUX 3 (fal, ~1 min)...')
        edited = core.fal_edit(load(out / 'grid_in.png'), prompt)
        save(out / 'grid_out.png', edited)
    log('3/4 extraindo máscaras...')
    visemes = core.make_visemes(ref, edited, det)
    for v, n in zip(visemes, core.NAMES):
        save(out / 'visemes' / f'{n}.png', v['image'])
        save(out / 'masks' / f'{n}.png', (v['mask'] * 255).round().astype(np.uint8))
    log('4/4 exportando (Phaser/Godot) e prévias...')
    files, _ = core.export_files(ref, visemes, det, a.atlas_name, a.godot_res_dir)
    core.write_files(files, out / 'export')
    save(out / 'preview_sheet.png', core.preview_sheet(visemes, det))
    by = {n: v['image'] for n, v in zip(core.NAMES, visemes)}; by[core.REST] = ref
    save_gif([by[n] for n in core.parse_seq(core.DEFAULT_SEQ)], out / 'preview_talk.gif', 110)
    save_gif([ref] + [v['image'] for v in visemes], out / 'preview_cycle.gif', 500)
    meta.update({'visemes': {n: {'image': f'visemes/{n}.png', 'mask': f'masks/{n}.png'} for n in core.NAMES},
                 'export': 'export/'})
    meta_p.write_text(json.dumps(meta, indent=2))
    log(f'pronto: {out.resolve()}')
    log('  export/ (atlas Phaser + .tres Godot + sprites + visemes.json), visemes/, masks/, previews')


def cmd_track(a):
    sd = Path(a.set)
    if not (sd / 'meta.json').is_file() or not (sd / 'visemes').is_dir():
        raise core.VisemirError(f'{sd} não tem visemas. Rode antes: cli.py generate ENTRADA -o {sd}')
    meta = json.loads((sd / 'meta.json').read_text()); det = meta['detection']
    ref = load(sd / 'ref.png')
    visemes = [{'image': load(sd / f'visemes/{n}.png'),
                'mask': np.asarray(Image.open(sd / f'masks/{n}.png').convert('L')).astype(np.float32) / 255} for n in core.NAMES]
    by = dict(zip(core.NAMES, visemes))
    seq = core.parse_seq(a.sequence)
    src = Source(a.video)
    if not src.is_video:
        raise core.VisemirError(f'{a.video} não é um vídeo')
    same = Path(meta.get('source', '')).resolve() == src.path.resolve()
    ref_index = a.ref_frame if a.ref_frame is not None else (meta.get('ref_frame', 0) if same else 0)
    ref_index = min(ref_index, src.n - 1)
    t0 = time.time(); done = [0]

    def tick():
        done[0] += 1
        if done[0] % 60 == 0: log(f'  tracking {done[0]}/{src.n}')
    Hs, ccs = core.track_all(src.get, src.n, ref, ref_index, core.make_tracker(ref, visemes, det), tick)
    write = close = None
    for k in range(src.n):
        fr = src.get(k); h, w = fr.shape[:2]
        if write is None:
            write, close = open_writer(a.output, w * 2 if a.debug else w, h, src.fps, a.video)
        name = core.viseme_at(seq, a.hold, k)
        o = fr if name == core.REST or ccs[k] < core.CC_LOW else core.composite(fr, by[name], Hs[k])  # perdido: não cola
        if a.debug:
            o = np.hstack([core.draw_track(fr, det, Hs[k], ccs[k], name), o])
        write(o)
    close()
    Path(a.output).with_suffix('.track.json').write_text(json.dumps(
        {'fps': src.fps, 'ref_frame': ref_index, 'homographies_ref_to_frame': [h.tolist() for h in Hs], 'confidence': ccs}))
    log(f'pronto: {a.output}  ({src.n} frames, {(time.time() - t0) / src.n * 1000:.0f} ms/frame, frame de referência {ref_index})')
    low = sum(c < core.CC_LOW for c in ccs)
    if low:
        log(f'aviso: {low} frame(s) com tracking de baixa confiança (cc<{core.CC_LOW}); confira com --debug')
    if a.gt:
        gt = json.loads(Path(a.gt).read_text()); mx0, my0, mx1, my1 = det['mouth']
        P = np.float32([[mx0, my0], [mx1, my0], [mx1, my1], [mx0, my1]])[None]
        e = [np.linalg.norm(cv2.transform(P, np.array(M))[0] - cv2.perspectiveTransform(P, Hh)[0], axis=1).max() for M, Hh in zip(gt, Hs)]
        log(f'erro vs ground truth na boca: média {np.mean(e):.2f}px  pior {np.max(e):.2f}px')


def cmd_demo(a):
    img = Source(a.input).get(0); H, W = img.shape[:2]
    write, close = open_writer(a.output, W, H, a.fps)
    tri = lambda t, p: 2 * abs((t / p) % 1 - 0.5) * 2 - 1  # onda triangular = zigzag
    amp = min(W, H); gt = []
    for t in range(a.frames):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 6 * np.sin(2 * np.pi * t / 75), 1 + 0.08 * np.sin(2 * np.pi * t / 60))
        M[:, 2] += (0.14 * amp * tri(t, 40), 0.07 * amp * tri(t, 26))
        write(cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(38, 35, 40)))
        gt.append(M.tolist())
    close()
    gp = Path(a.output).with_suffix('.gt.json'); gp.write_text(json.dumps(gt))
    log(f'pronto: {a.output}  (movimento real em {gp})')


def main(argv=None):
    p = argparse.ArgumentParser(prog='visemir', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest='cmd', required=True)
    sp.add_parser('doctor', help='verifica dependências e chaves').set_defaults(fn=cmd_doctor)

    loc = argparse.ArgumentParser(add_help=False)
    loc.add_argument('--frame', type=int, default=0, help='frame inicial onde procurar a boca (vídeo)')
    loc.add_argument('--retry-step', type=int, default=12, help='se não achar, pula N frames (padrão 12)')
    loc.add_argument('--max-attempts', type=int, default=5, help='máximo de frames tentados (padrão 5)')
    loc.add_argument('--mode', choices=core.DETECT_MODES, default='auto', help='auto = boca -> nariz/queixo -> rosto')
    loc.add_argument('--face-frac', type=float, default=0.9, help='tamanho do recorte / altura do rosto (padrão 0.9)')

    d = sp.add_parser('detect', parents=[loc], help='acha a boca e a área de recorte')
    d.add_argument('input'); d.add_argument('--debug', metavar='PNG', help='salva imagem com as caixas')
    d.set_defaults(fn=cmd_detect)

    sty = argparse.ArgumentParser(add_help=False)
    sty.add_argument('--style', default='anime', help='estilo citado no prompt')

    pr = sp.add_parser('prepare', parents=[loc, sty], help='só grid + prompt (para modelo local)')
    pr.add_argument('input'); pr.add_argument('-o', '--output', required=True)
    pr.set_defaults(fn=cmd_prepare)

    g = sp.add_parser('generate', parents=[loc, sty], help='gera visemas + exportação')
    g.add_argument('input'); g.add_argument('-o', '--output', required=True, help='pasta de saída')
    g.add_argument('--edited', metavar='PNG', help='grid já editado (modelo local); não chama o fal')
    g.add_argument('--atlas-name', default='visemes_atlas')
    g.add_argument('--godot-res-dir', default='res://', help='pasta do atlas no projeto Godot')
    g.set_defaults(fn=cmd_generate)

    t = sp.add_parser('track', help='cola os visemas num vídeo')
    t.add_argument('video'); t.add_argument('--set', required=True, help='pasta criada pelo generate')
    t.add_argument('-o', '--output', required=True, help='vídeo de saída (.mp4)')
    t.add_argument('--sequence', default=core.DEFAULT_SEQ, help=f'ordem das bocas ({", ".join(core.NAMES)}, {core.REST})')
    t.add_argument('--hold', type=int, default=4, help='frames por boca (padrão 4)')
    t.add_argument('--ref-frame', type=int, help='frame do vídeo que é a referência (padrão: o do generate)')
    t.add_argument('--debug', action='store_true', help='mostra o tracking lado a lado')
    t.add_argument('--gt', help=argparse.SUPPRESS)
    t.set_defaults(fn=cmd_track)

    m = sp.add_parser('demo', help='vídeo de teste em zigzag')
    m.add_argument('input'); m.add_argument('-o', '--output', required=True)
    m.add_argument('--frames', type=int, default=150); m.add_argument('--fps', type=float, default=30)
    m.set_defaults(fn=cmd_demo)

    a = p.parse_args(argv)
    for k, lo in (('hold', 1), ('retry_step', 1), ('max_attempts', 1), ('frame', 0), ('frames', 1)):
        if getattr(a, k, lo) is not None and getattr(a, k, lo) < lo:
            p.error(f'--{k.replace("_", "-")} precisa ser >= {lo}')
    if getattr(a, 'face_frac', 1) <= 0:
        p.error('--face-frac precisa ser > 0')
    try:
        return a.fn(a) or 0
    except core.VisemirError as e:
        print(f'erro: {e}', file=sys.stderr); return 1
    except KeyboardInterrupt:
        print('interrompido.', file=sys.stderr); return 130


if __name__ == '__main__':
    sys.exit(main())
