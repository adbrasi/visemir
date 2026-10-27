"""Nodes do ComfyUI: Prepare Grid -> Edit Grid (fal ou modelo local) -> Apply To Video e/ou Export Sprites / Atlas."""
import json

import numpy as np
import torch

from . import visemir_core as core

try:
    from comfy.utils import ProgressBar
except ImportError:  # fora do ComfyUI
    ProgressBar = None
try:
    from comfy.model_management import throw_exception_if_processing_interrupted as check_interrupt
except ImportError:
    def check_interrupt(): pass


def frame_np(video, i):
    return (video[i, ..., :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


def to_tensor(arrs):
    return torch.from_numpy(np.stack(arrs)).float() / 255


def progress(total):
    if ProgressBar is None:
        return None
    pb = ProgressBar(total)
    return lambda: pb.update(1)


def run(fn):
    try:
        return fn()
    except core.VisemirError as e:
        raise RuntimeError(f'Visemir: {e}') from None


class VisemirPrepareGrid:
    """Acha a boca num frame do vídeo (Moondream) e monta o grid 3x2 para a edição."""
    CATEGORY = 'visemir'
    FUNCTION = 'prepare'
    RETURN_TYPES = ('IMAGE', 'VISEMIR', 'IMAGE', 'INT')
    RETURN_NAMES = ('grid', 'visemir_data', 'detection_preview', 'ref_frame')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'video': ('IMAGE',),
            'moondream_api_key': ('STRING', {'default': '', 'tooltip': 'vazio = usa a variável de ambiente MOONDREAM_API_KEY'}),
            'start_frame': ('INT', {'default': 0, 'min': 0, 'max': 1_000_000, 'tooltip': 'frame onde começa a procurar a boca'}),
            'retry_step': ('INT', {'default': 12, 'min': 1, 'max': 10_000, 'tooltip': 'se não achar, pula esta quantidade de frames'}),
            'max_attempts': ('INT', {'default': 5, 'min': 1, 'max': 100, 'tooltip': 'quantos frames tentar no máximo'}),
            'detect_mode': (core.DETECT_MODES, {'default': 'auto', 'tooltip': 'auto = boca -> nariz/queixo -> rosto'}),
            'face_frac': ('FLOAT', {'default': 0.9, 'min': 0.3, 'max': 2.0, 'step': 0.05,
                                    'tooltip': 'tamanho do recorte em relação à altura do rosto'}),
        }}

    def prepare(self, video, moondream_api_key, start_frame, retry_step, max_attempts, detect_mode, face_frac):
        def go():
            idx, ref, det = core.find_reference(lambda i: frame_np(video, i), video.shape[0], start_frame, retry_step,
                                                max_attempts, moondream_api_key, detect_mode, face_frac)
            core.log(f'frame {idx}: método {det["method"]}, boca {det["mouth"]}, recorte {det["crop"]}')
            data = {'ref': ref, 'ref_index': idx, 'det': det}
            return (to_tensor([core.build_grid(ref, det)]), data, to_tensor([core.draw_detection(ref, det)]), idx)
        return run(go)


class VisemirEditGrid:
    """Edita o grid no FLUX 3 (fal) ou só entrega prompt + tamanho exato para rodar num modelo local."""
    CATEGORY = 'visemir'
    FUNCTION = 'edit'
    RETURN_TYPES = ('IMAGE', 'STRING', 'INT', 'INT')
    RETURN_NAMES = ('edited_grid', 'prompt', 'width', 'height')
    BACKENDS = ['fal (FLUX 3)', 'local (prompt only)']

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'grid': ('IMAGE',),
            'backend': (cls.BACKENDS, {'default': cls.BACKENDS[0]}),
            'fal_api_key': ('STRING', {'default': '', 'tooltip': 'vazio = usa a variável de ambiente FAL_API_KEY'}),
            'style': ('STRING', {'default': 'anime', 'tooltip': 'estilo citado no prompt'}),
        }}

    def edit(self, grid, backend, fal_api_key, style):
        def go():
            g = frame_np(grid, 0); prompt = core.build_prompt(style)
            if backend == self.BACKENDS[1]:
                # local: devolve o grid original + prompt + tamanho (múltiplos de 16) para o latente
                return (grid[:1], prompt, g.shape[1], g.shape[0])
            out = core.fal_edit(g, prompt, fal_api_key)
            return (to_tensor([out]), prompt, out.shape[1], out.shape[0])
        return run(go)


class VisemirApplyToVideo:
    """Recebe o grid editado (fal ou local), extrai as 6 bocas e cola no vídeo inteiro com tracking."""
    CATEGORY = 'visemir'
    FUNCTION = 'apply'
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'MASK', 'IMAGE', 'STRING')
    RETURN_NAMES = ('video', 'visemes', 'viseme_masks', 'preview_sheet', 'tracking_json')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'video': ('IMAGE',),
            'edited_grid': ('IMAGE',),
            'visemir_data': ('VISEMIR',),
            'sequence': ('STRING', {'default': core.DEFAULT_SEQ,
                                    'tooltip': f'ordem das bocas: {", ".join(core.NAMES)}, {core.REST} (boca original)'}),
            'hold': ('INT', {'default': 4, 'min': 1, 'max': 1000, 'tooltip': 'frames que cada boca fica na tela'}),
            'draw_debug': ('BOOLEAN', {'default': False, 'tooltip': 'desenha a região rastreada e a confiança'}),
        }}

    def apply(self, video, edited_grid, visemir_data, sequence, hold, draw_debug):
        def go():
            ref, det, ref_index = visemir_data['ref'], visemir_data['det'], visemir_data['ref_index']
            seq = core.parse_seq(sequence)
            visemes = core.make_visemes(ref, frame_np(edited_grid, 0), det)
            by = dict(zip(core.NAMES, visemes))
            n = video.shape[0]
            tracker = core.make_tracker(ref, visemes, det)
            tick = progress(n * 2)

            def get(i):
                check_interrupt(); return frame_np(video, i)
            Hs, ccs = core.track_all(get, n, ref, ref_index, tracker, tick)
            out = torch.empty((n,) + tuple(video.shape[1:3]) + (3,), dtype=torch.float32)
            for k in range(n):
                check_interrupt()
                fr = frame_np(video, k); name = core.viseme_at(seq, hold, k)
                o = fr if name == core.REST or ccs[k] < core.CC_LOW else core.composite(fr, by[name], Hs[k])  # perdido: não cola
                if draw_debug:
                    o = core.draw_track(o, det, Hs[k], ccs[k], name)
                out[k] = torch.from_numpy(o).float() / 255
                if tick: tick()
            low = sum(c < core.CC_LOW for c in ccs)
            if low:
                core.log(f'aviso: {low} frame(s) com tracking de baixa confiança; ligue draw_debug para conferir')
            vis = to_tensor([v['image'] for v in visemes])
            masks = torch.from_numpy(np.stack([v['mask'] for v in visemes])).float()
            sheet = to_tensor([core.preview_sheet(visemes, det)])
            info = json.dumps({'ref_frame': ref_index, 'detection': det, 'order': core.NAMES,
                               'homographies_ref_to_frame': [h.tolist() for h in Hs], 'confidence': ccs})
            return (out, vis, masks, sheet, info)
        return run(go)


class VisemirExportSprites:
    """Sem vídeo: exporta as bocas recortadas + coordenadas exatas (atlas Phaser, AtlasTexture Godot, PNGs, JSON)."""
    CATEGORY = 'visemir'
    FUNCTION = 'export'
    OUTPUT_NODE = True
    RETURN_TYPES = ('IMAGE', 'MASK', 'IMAGE', 'MASK', 'STRING', 'STRING')
    RETURN_NAMES = ('atlas', 'atlas_alpha', 'sprites', 'sprites_alpha', 'coords_json', 'output_dir')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'edited_grid': ('IMAGE',),
            'visemir_data': ('VISEMIR',),
            'folder_prefix': ('STRING', {'default': 'visemir/personagem', 'tooltip': 'pasta dentro de ComfyUI/output'}),
            'atlas_name': ('STRING', {'default': 'visemes_atlas'}),
            'godot_res_dir': ('STRING', {'default': 'res://', 'tooltip': 'pasta do atlas no projeto Godot (para os .tres)'}),
        }}

    def export(self, edited_grid, visemir_data, folder_prefix, atlas_name, godot_res_dir):
        def go():
            import os
            ref, det = visemir_data['ref'], visemir_data['det']
            visemes = core.make_visemes(ref, frame_np(edited_grid, 0), det)
            files, atlas = core.export_files(ref, visemes, det, atlas_name.strip() or 'visemes_atlas', godot_res_dir)
            try:
                import folder_paths
                root = folder_paths.get_output_directory()
            except ImportError:
                root = os.path.abspath('output')
            base = os.path.join(root, folder_prefix.strip().strip('/\\') or 'visemir')
            n = 1
            while os.path.exists(f'{base}_{n:04d}'):
                n += 1
            out = str(core.write_files(files, f'{base}_{n:04d}'))
            core.log(f'exportado em {out}')
            # sprites em tamanho da base (boca já no lugar), para ver/compor dentro do ComfyUI
            sp = [v['image'] for v in visemes]
            return (to_tensor([atlas[..., :3]]), torch.from_numpy(atlas[..., 3].astype(np.float32) / 255)[None],
                    to_tensor(sp), torch.from_numpy(np.stack([v['mask'] for v in visemes])).float(),
                    files['visemes.json'].decode(), out)
        return run(go)


NODE_CLASS_MAPPINGS = {
    'VisemirPrepareGrid': VisemirPrepareGrid,
    'VisemirEditGrid': VisemirEditGrid,
    'VisemirApplyToVideo': VisemirApplyToVideo,
    'VisemirExportSprites': VisemirExportSprites,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'VisemirPrepareGrid': 'Visemir · Prepare Grid',
    'VisemirEditGrid': 'Visemir · Edit Grid (fal / local)',
    'VisemirApplyToVideo': 'Visemir · Apply To Video',
    'VisemirExportSprites': 'Visemir · Export Sprites / Atlas',
}
