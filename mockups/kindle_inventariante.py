"""Render do mockup de lançamento: a capa de "A Inventariante" na tela de um Kindle.

Medidas do aparelho seguem o Kindle 11a geração (corpo 157,8 x 108,6 mm, tela de
6" com 90,7 x 122,6 mm de área ativa). Os bezels laterais são derivados por
construção (BW = SW + 2*SIDE) para ficarem exatamente simétricos. A razão
queixo/moldura superior fica em ~2,9x, que é a do aparelho real.

A capa entra em "cover": preenche a área ativa e é cortada ~4% no topo e na base.
O corte não alcança o bloco tipográfico, que começa a ~14% da altura.

Uso:
    python3 kindle_inventariante.py capa.png saida.png
"""

import math
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONTB = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

SW, SH = 735, 994              # área ativa da tela
SIDE, TOP, CHIN = 73, 73, 211
BW, BH = SW + 2 * SIDE, TOP + SH + CHIN
RAD = 75
CANVAS = 1600
BACK_SCALE = 0.94              # traseiro menor + desfocado: profundidade
BACK_DX, BACK_DY = -0.56, -0.085

rng = np.random.default_rng(11)


def rrect(w, h, r, ss=4, outline=False, width=2):
    m = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(m)
    if outline:
        d.rounded_rectangle([1, 1, w * ss - 2, h * ss - 2], radius=r * ss,
                            outline=255, width=width * ss)
    else:
        d.rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=r * ss, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def vgrad(w, h, c0, c1, gamma=1.0):
    """Gradiente vertical com dither, para não posterizar nas áreas escuras."""
    t = (np.linspace(0, 1, h) ** gamma)[:, None]
    a = np.array(c0)[None, None, :] + (np.array(c1) - np.array(c0))[None, None, :] * t[:, :, None]
    a = np.repeat(a, w, axis=1) + rng.uniform(-1.6, 1.6, (h, w, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def tracked(d, cx, y, text, font, fill, tr):
    wid = sum(d.textlength(c, font=font) for c in text) + tr * (len(text) - 1)
    x = cx - wid / 2
    for c in text:
        d.text((x, y), c, font=font, fill=fill)
        x += d.textlength(c, font=font) + tr


def eink(cover_path):
    """A capa como a tela e-ink a exibe.

    A ordem importa: a frontlight e a vinheta entram ANTES do mapeamento de
    faixa dinâmica. Somadas depois, elas estouravam o teto e produziam branco
    puro, que papel eletrônico nenhum alcança. O grão entra depois do
    redimensionamento e não há blur, para a tela não sair mais macia que a arte.
    """
    cov = Image.open(cover_path).convert("RGB")
    k = SW / cov.width
    cov = cov.resize((SW, round(cov.height * k)), Image.LANCZOS)   # escala uniforme
    off = (cov.height - SH) // 2
    cov = cov.crop((0, off, SW, off + SH))                         # corte só na vertical

    a = np.asarray(cov).astype(np.float32)
    a = a * 0.94 + a.mean(axis=2, keepdims=True) * 0.06            # dessaturação leve
    yy = np.linspace(0, 1, SH)[:, None, None]
    xx = np.linspace(0, 1, SW)[None, :, None]
    a = a + 14.0 * (yy ** 2.6)                                     # LEDs na base
    rad = np.sqrt(((xx - 0.5) * 0.74) ** 2 + (yy - 0.5) ** 2) / 0.62
    a = np.clip(a - 12.0 * np.clip(rad, 0, 1) ** 2.4, 0, 255)
    a = 26.0 + a * (228.0 - 26.0) / 255.0                          # teto garantido: 228
    a = a * np.array([1.010, 1.0, 0.978])[None, None, :]           # papel morno
    a = a + rng.normal(0, 2.2, a.shape)                            # grão de partícula
    scr = Image.fromarray(np.clip(a, 0, 238).astype(np.uint8))

    sh = Image.new("L", (SW, SH), 0)
    ImageDraw.Draw(sh).rectangle([0, 0, SW - 1, SH - 1], outline=255, width=11)
    scr = Image.composite(Image.new("RGB", (SW, SH), (10, 10, 12)), scr,
                          sh.filter(ImageFilter.GaussianBlur(9)).point(lambda v: int(v * 0.42)))

    sn = Image.new("L", (SW, SH), 0)                               # reflexo de vidro
    ImageDraw.Draw(sn).polygon([(-60, 0), (SW * 0.58, 0), (-60, SH * 0.66)], fill=255)
    return Image.composite(Image.new("RGB", (SW, SH), (255, 255, 255)), scr,
                           sn.filter(ImageFilter.GaussianBlur(95)).point(lambda v: int(v * 0.055)))


def hlight(dev, amp, sb):
    """Variação horizontal + reflexo alongado.

    Sem isto cada linha do corpo é um valor único repetido na largura inteira, e
    a peça lê como shape vetorial chapado em vez de objeto.
    """
    x = np.linspace(0, 1, BW)
    prof = amp * (0.5 - 0.5 * np.cos(2 * np.pi * x)) ** 0.7
    lay = np.clip(np.repeat(prof[None, :], BH, axis=0), 0, 255).astype(np.uint8)
    dev.paste(Image.new("RGB", (BW, BH), (190, 190, 200)), (0, 0), Image.fromarray(lay))
    g = Image.new("L", (BW, BH), 0)
    ImageDraw.Draw(g).ellipse([int(BW * 0.06), int(-BH * 0.10),
                               int(BW * 0.30), int(BH * 1.10)], fill=255)
    dev.paste(Image.new("RGB", (BW, BH), (210, 210, 220)), (0, 0),
              g.filter(ImageFilter.GaussianBlur(70)).point(lambda v: int(v * sb)))


def edges(dev, lo_top, lo_bot, left_rim=0.0):
    m = rrect(BW, BH, RAD, outline=True, width=3).filter(ImageFilter.GaussianBlur(1.3))
    m = ImageChops.multiply(m, vgrad(BW, BH, (255, 255, 255), (lo_top,) * 3, 0.75).convert("L"))
    dev.paste(Image.new("RGB", (BW, BH), (168, 168, 177)), (0, 0), m.point(lambda v: int(v * 0.60)))
    b = Image.new("L", (BW, BH), 0)
    ImageDraw.Draw(b).rounded_rectangle([6, BH - 14, BW - 7, BH - 3], radius=RAD // 2, fill=255)
    dev.paste(Image.new("RGB", (BW, BH), (120, 120, 128)), (0, 0),
              b.filter(ImageFilter.GaussianBlur(4)).point(lambda v: int(v * lo_bot)))
    if left_rim:
        # Luz de recorte: sem ela o frontal se funde com a massa preta do traseiro.
        r = Image.new("L", (BW, BH), 0)
        ImageDraw.Draw(r).rounded_rectangle([2, RAD // 2, 7, BH - RAD // 2], radius=3, fill=255)
        dev.paste(Image.new("RGB", (BW, BH), (196, 196, 205)), (0, 0),
                  r.filter(ImageFilter.GaussianBlur(2.2)).point(lambda v: int(v * left_rim)))


def front(scr):
    dev = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    dev.paste(vgrad(BW, BH, (50, 50, 53), (20, 20, 23), 1.15), (0, 0), rrect(BW, BH, RAD))
    dev.paste(scr, (SIDE, TOP))
    hlight(dev, 26, 0.09)
    edges(dev, 40, 0.42, left_rim=0.75)
    d = ImageDraw.Draw(dev)
    d.rectangle([SIDE - 1, TOP - 1, SIDE + SW, TOP + SH], outline=(8, 8, 10, 255), width=2)
    tracked(d, BW / 2, TOP + SH + CHIN / 2 - 20, "kindle",
            ImageFont.truetype(FONT, 38), (104, 104, 111, 255), 3.4)
    dev.putalpha(rrect(BW, BH, RAD))
    return dev


def smile(dr, cx, cy, w, col, ss):
    """Sorriso da Amazon com massa variável e seta integrada ao arco.

    Um arco de espessura constante lê como fio solto, não como a marca.
    """
    rx, ry = w / 2.0, w * 0.30
    a0, a1 = math.radians(190), math.radians(357)
    n = 90
    pts = []
    for i in range(n + 1):
        t = a0 + (a1 - a0) * i / n
        pts.append((cx + rx * math.cos(t), cy - ry * math.sin(t)))
    for i in range(n, -1, -1):
        t = a0 + (a1 - a0) * i / n
        f = 0.55 + 0.30 * math.sin(math.pi * i / n)                # afina nas pontas
        pts.append((cx + rx * f * math.cos(t), cy - ry * f * math.sin(t)))
    dr.polygon([(x * ss, y * ss) for x, y in pts], fill=col)
    tx, ty = cx + rx * math.cos(a1), cy - ry * math.sin(a1)
    dr.polygon([((tx + w * 0.10) * ss, (ty - w * 0.10) * ss),
                ((tx - w * 0.06) * ss, (ty + w * 0.015) * ss),
                ((tx + w * 0.02) * ss, (ty + w * 0.09) * ss)], fill=col)


def back():
    dev = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    body = np.asarray(vgrad(BW, BH, (40, 40, 43), (16, 16, 19), 1.25)).astype(np.float32)
    body += rng.normal(0, 1.5, body.shape)                         # acabamento fosco
    dev.paste(Image.fromarray(np.clip(body, 0, 255).astype(np.uint8)), (0, 0), rrect(BW, BH, RAD))

    ss = 3
    arc = Image.new("L", (BW * ss, BH * ss), 0)                    # chanfro só no alto
    ImageDraw.Draw(arc).arc([int(BW * 0.05 * ss), int(-BH * 0.06 * ss),
                             int(BW * 1.28 * ss), int(BH * 0.26 * ss)],
                            start=10, end=100, fill=255, width=int(8 * ss))
    dev.paste(Image.new("RGB", (BW, BH), (120, 120, 128)), (0, 0),
              arc.resize((BW, BH), Image.LANCZOS)
                 .filter(ImageFilter.GaussianBlur(9)).point(lambda v: int(v * 0.30)))

    # Logo a 40% da largura: centralizado ele ficaria escondido atrás do frontal.
    # Tom próximo do corpo, para ler como relevo e não como adesivo.
    lx, ly, lw = BW * 0.40, BH * 0.55, BW * 0.23
    lg = Image.new("L", (BW * ss, BH * ss), 0)
    smile(ImageDraw.Draw(lg), lx, ly, lw, 255, ss)
    dev.paste(Image.new("RGB", (BW, BH), (56, 56, 62)), (0, 0),
              lg.resize((BW, BH), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.8)))
    d = ImageDraw.Draw(dev)
    f = ImageFont.truetype(FONTB, int(lw * 0.28))
    tw = d.textlength("amazon", font=f)
    d.text((lx - tw / 2, ly - lw * 0.40), "amazon", font=f, fill=(55, 55, 61, 255))

    hlight(dev, 20, 0.07)
    edges(dev, 30, 0.30)
    dev.putalpha(rrect(BW, BH, RAD))
    return dev


def compose(cover_path, out_path):
    scr = eink(cover_path)
    canvas = np.full((CANVAS, CANVAS, 3), 255.0, dtype=np.float32)

    bw2, bh2 = round(BW * BACK_SCALE), round(BH * BACK_SCALE)
    bx, by = round(BW * BACK_DX), round(BH * BACK_DY)
    minx, miny = min(0, bx), min(0, by)
    maxx, maxy = max(BW, bx + bw2), max(BH, by + bh2)
    ox = (CANVAS - (maxx - minx)) // 2 - minx
    oy = (CANVAS - (maxy - miny)) // 2 - miny
    FX, FY, BX, BY = ox, oy, bx + ox, by + oy

    def shade(pos, size, blur, dx, dy, strength):
        """Sombra multiplicativa.

        Colar cinza por cima parecia sombra sobre o fundo branco, mas CLAREAVA o
        aparelho escuro de trás. Multiplicar escurece os dois corretamente.
        """
        nonlocal canvas
        lay = Image.new("L", (CANVAS, CANVAS), 0)
        lay.paste(rrect(size[0], size[1], round(RAD * size[0] / BW)),
                  (pos[0] + dx, pos[1] + dy))
        m = np.asarray(lay.filter(ImageFilter.GaussianBlur(blur))).astype(np.float32) / 255.0
        canvas *= (1.0 - strength * m)[:, :, None]

    def put(img, pos):
        nonlocal canvas
        arr = np.asarray(img).astype(np.float32)
        al = arr[:, :, 3:4] / 255.0
        x, y = pos
        h, w = arr.shape[:2]
        canvas[y:y + h, x:x + w, :] = canvas[y:y + h, x:x + w, :] * (1 - al) + arr[:, :, :3] * al

    # Luz vinda de cima-esquerda: as sombras caem para baixo-direita.
    shade((BX, BY), (bw2, bh2), 40, 14, 32, 0.34)
    shade((BX, BY), (bw2, bh2), 12, 4, 9, 0.32)                    # contato do traseiro
    put(back().resize((bw2, bh2), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.1)), (BX, BY))
    shade((FX, FY), (BW, BH), 44, -34, 16, 0.34)                   # frontal SOBRE o traseiro
    shade((FX, FY), (BW, BH), 36, 16, 28, 0.36)
    shade((FX, FY), (BW, BH), 12, 4, 8, 0.30)
    put(front(scr), (FX, FY))

    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    out.save(out_path)
    return out


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "capa.png"
    dst = sys.argv[2] if len(sys.argv) > 2 else "a-inventariante-kindle.png"
    compose(src, dst)
    print("gravado:", dst)
