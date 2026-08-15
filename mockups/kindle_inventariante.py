"""Render do mockup de lançamento: a capa de "A Inventariante" na tela de um Kindle.

Medidas do aparelho seguem o Kindle 11a geração (corpo 157,8 x 108,6 mm, tela de
6" com 90,7 x 122,6 mm de área ativa). Os bezels laterais são derivados por
construção (BW = SW + 2*SIDE) para ficarem exatamente simétricos.

A capa entra em "cover": preenche a área ativa e é cortada ~4% no topo e na base.
O corte não alcança o bloco tipográfico, que começa a 14,5% da altura.

Uso:
    python3 kindle_inventariante.py capa.png saida.png
"""

import math
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

SW, SH = 735, 994          # área ativa da tela
SIDE, TOP, CHIN = 73, 73, 211
BW, BH = SW + 2 * SIDE, TOP + SH + CHIN
RAD = 75
CANVAS = 1600
BACK_SCALE = 0.94          # traseiro menor: dá profundidade
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
    a = np.repeat(a, w, axis=1) + rng.uniform(-0.6, 0.6, (h, w, 3))
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
    faixa dinâmica, senão o realce estoura o teto do e-ink e aparece branco
    puro — que papel eletrônico nenhum alcança. O grão entra depois do
    redimensionamento e não há blur, para a tela não ficar mais macia que a arte.
    """
    cov = Image.open(cover_path).convert("RGB")
    k = SW / cov.width
    cov = cov.resize((SW, round(cov.height * k)), Image.LANCZOS)
    off = (cov.height - SH) // 2
    cov = cov.crop((0, off, SW, off + SH))

    a = np.asarray(cov).astype(np.float32)
    a = a * 0.90 + a.mean(axis=2, keepdims=True) * 0.10
    yy = np.linspace(0, 1, SH)[:, None, None]
    xx = np.linspace(0, 1, SW)[None, :, None]
    a = a + 14.0 * (yy ** 2.6)                                   # LEDs na base
    rad = np.sqrt(((xx - 0.5) * 0.74) ** 2 + (yy - 0.5) ** 2) / 0.62
    a = np.clip(a - 9.0 * np.clip(rad, 0, 1) ** 2.4, 0, 255)
    a = 26.0 + a * (228.0 - 26.0) / 255.0                        # teto garantido
    a = a * np.array([1.010, 1.0, 0.978])[None, None, :]         # papel morno
    a = a + rng.normal(0, 2.2, a.shape)                          # grão de partícula
    scr = Image.fromarray(np.clip(a, 0, 238).astype(np.uint8))

    sh = Image.new("L", (SW, SH), 0)
    ImageDraw.Draw(sh).rectangle([0, 0, SW - 1, SH - 1], outline=255, width=8)
    return Image.composite(Image.new("RGB", (SW, SH), (12, 12, 14)), scr,
                           sh.filter(ImageFilter.GaussianBlur(7)).point(lambda v: int(v * 0.34)))


def edges(dev, lo_top, lo_bot):
    """Aresta iluminada em volta e realce na base, para o corpo ter espessura."""
    m = rrect(BW, BH, RAD, outline=True, width=3).filter(ImageFilter.GaussianBlur(1.3))
    m = ImageChops.multiply(m, vgrad(BW, BH, (255, 255, 255), (lo_top,) * 3, 0.75).convert("L"))
    dev.paste(Image.new("RGB", (BW, BH), (168, 168, 177)), (0, 0), m.point(lambda v: int(v * 0.60)))
    b = Image.new("L", (BW, BH), 0)
    ImageDraw.Draw(b).rounded_rectangle([6, BH - 14, BW - 7, BH - 3], radius=RAD // 2, fill=255)
    dev.paste(Image.new("RGB", (BW, BH), (120, 120, 128)), (0, 0),
              b.filter(ImageFilter.GaussianBlur(4)).point(lambda v: int(v * lo_bot)))


def front(scr):
    dev = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    dev.paste(vgrad(BW, BH, (50, 50, 53), (20, 20, 23), 1.15), (0, 0), rrect(BW, BH, RAD))
    dev.paste(scr, (SIDE, TOP))
    edges(dev, 40, 0.42)
    d = ImageDraw.Draw(dev)
    d.rectangle([SIDE - 1, TOP - 1, SIDE + SW, TOP + SH], outline=(8, 8, 10, 255), width=2)
    tracked(d, BW / 2, TOP + SH + CHIN / 2 - 24, "kindle",
            ImageFont.truetype(FONT, 44), (126, 126, 133, 255), 4.2)
    dev.putalpha(rrect(BW, BH, RAD))
    return dev


def back():
    dev = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    body = np.asarray(vgrad(BW, BH, (43, 43, 46), (18, 18, 21), 1.25)).astype(np.float32)
    body += rng.normal(0, 1.5, body.shape)                       # acabamento fosco
    dev.paste(Image.fromarray(np.clip(body, 0, 255).astype(np.uint8)), (0, 0), rrect(BW, BH, RAD))

    ss = 3
    arc = Image.new("L", (BW * ss, BH * ss), 0)
    ImageDraw.Draw(arc).arc([int(BW * 0.05 * ss), int(BH * 0.02 * ss),
                             int(BW * 1.28 * ss), int(BH * 0.40 * ss)],
                            start=8, end=104, fill=255, width=int(9 * ss))
    dev.paste(Image.new("RGB", (BW, BH), (132, 132, 140)), (0, 0),
              arc.resize((BW, BH), Image.LANCZOS)
                 .filter(ImageFilter.GaussianBlur(8)).point(lambda v: int(v * 0.38)))

    # Logo deslocado para 40% da largura: centralizado ele ficaria escondido
    # atrás do aparelho da frente.
    lw = BW * 0.34
    lx, ly = BW * 0.40, BH * 0.52
    lg = Image.new("L", (BW * ss, BH * ss), 0)
    dl = ImageDraw.Draw(lg)
    dl.arc([int((lx - lw / 2) * ss), int((ly - lw * 0.40) * ss),
            int((lx + lw / 2) * ss), int((ly + lw * 0.40) * ss)],
           start=16, end=152, fill=255, width=int(8 * ss))
    tx = lx + lw / 2 * math.cos(math.radians(16))
    ty = ly + lw * 0.40 * math.sin(math.radians(16))
    dl.polygon([((tx + 20) * ss, (ty - 18) * ss), ((tx - 10) * ss, (ty + 3) * ss),
                ((tx + 7) * ss, (ty + 19) * ss)], fill=255)
    dev.paste(Image.new("RGB", (BW, BH), (78, 78, 84)), (0, 0),
              lg.resize((BW, BH), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.3)))

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

    def shade(pos, size, blur, dy, strength):
        """Sombra multiplicativa.

        Colar cinza por cima parecia sombra sobre o fundo branco, mas CLAREAVA o
        aparelho escuro de trás. Multiplicar escurece os dois corretamente.
        """
        nonlocal canvas
        lay = Image.new("L", (CANVAS, CANVAS), 0)
        lay.paste(rrect(size[0], size[1], round(RAD * size[0] / BW)), (pos[0], pos[1] + dy))
        m = np.asarray(lay.filter(ImageFilter.GaussianBlur(blur))).astype(np.float32) / 255.0
        canvas *= (1.0 - strength * m)[:, :, None]

    def put(img, pos):
        nonlocal canvas
        arr = np.asarray(img).astype(np.float32)
        al = arr[:, :, 3:4] / 255.0
        x, y = pos
        h, w = arr.shape[:2]
        canvas[y:y + h, x:x + w, :] = canvas[y:y + h, x:x + w, :] * (1 - al) + arr[:, :, :3] * al

    bk = back().resize((bw2, bh2), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.1))
    shade((BX, BY), (bw2, bh2), 38, 30, 0.34)
    shade((BX, BY), (bw2, bh2), 13, 7, 0.22)       # contato
    put(bk, (BX, BY))
    shade((FX, FY), (BW, BH), 34, 26, 0.38)        # cai também sobre o traseiro
    shade((FX, FY), (BW, BH), 12, 7, 0.26)
    put(front(scr), (FX, FY))

    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    out.save(out_path)
    return out


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "capa.png"
    dst = sys.argv[2] if len(sys.argv) > 2 else "a-inventariante-kindle.png"
    compose(src, dst)
    print("gravado:", dst)
