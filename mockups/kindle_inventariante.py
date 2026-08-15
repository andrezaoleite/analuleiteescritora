"""Render do mockup de lançamento: a capa de "A Inventariante" na tela de um Kindle.

A tela é dimensionada para que a capa apareça em resolução nativa (774 px de
altura), sem upscale — por isso a tipografia continua nítida no arquivo final.
As medidas do aparelho vêm do Kindle 11a geração (157,8 x 108,6 mm de corpo,
tela de 6" com 90,7 x 122,6 mm de área ativa), convertidas para pixels.

Uso:
    python3 kindle_inventariante.py capa.jpg saida.png
"""

import math
import random
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

# Geometria: mm -> px, ancorada na altura nativa da capa.
S = 774 / 122.6
SW, SH = round(90.7 * S), 774
BW, BH = round(108.6 * S), round(157.8 * S)
SIDE = (BW - SW) // 2
TOP = round(9.0 * S)
RAD = round(9.2 * S)


def rrect(w, h, r, ss=4, outline=False, width=2):
    """Retângulo arredondado com antialiasing por supersampling."""
    m = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(m)
    if outline:
        d.rounded_rectangle([1, 1, w * ss - 2, h * ss - 2], radius=r * ss,
                            outline=255, width=width * ss)
    else:
        d.rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=r * ss, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def vgrad(w, h, c0, c1, gamma=1.0):
    g = Image.new("RGB", (1, h))
    for y in range(h):
        t = (y / (h - 1)) ** gamma
        g.putpixel((0, y), tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3)))
    return g.resize((w, h), Image.BICUBIC)


def tracked(draw, cx, y, text, font, fill, tr):
    """Texto centrado com entreletra, para a marca na moldura inferior."""
    wid = sum(draw.textlength(c, font=font) for c in text) + tr * (len(text) - 1)
    x = cx - wid / 2
    for c in text:
        draw.text((x, y), c, font=font, fill=fill)
        x += draw.textlength(c, font=font) + tr


def eink(cover):
    """Aplica o comportamento da tela e-ink sobre a capa.

    A capa entra em "contain" (a proporção dela é mais estreita que a da tela),
    então sobra a margem de papel do próprio e-ink nas laterais — que é o que um
    Kindle real faz, sem cortar a arte.
    """
    scr = Image.new("RGB", (SW, SH), (240, 238, 233))
    scr.paste(cover, ((SW - cover.width) // 2, (SH - cover.height) // 2))

    scr = Image.blend(scr, scr.convert("L").convert("RGB"), 0.10)
    # e-ink não alcança preto nem branco puros: comprime a faixa dinâmica.
    scr = scr.point([int(24 + v * (241 - 24) / 255) for v in range(256)] * 3)
    r, g, b = scr.split()
    scr = Image.merge("RGB", (r.point(lambda v: min(255, int(v * 1.010))), g,
                              b.point(lambda v: int(v * 0.978))))

    # Frontlight: LEDs na borda inferior, caindo suavemente para cima.
    fl = Image.new("L", (SW, SH), 0)
    fp = fl.load()
    for y in range(SH):
        base = int(15 * ((y / (SH - 1)) ** 2.6))
        for x in range(SW):
            fp[x, y] = base
    fl = fl.filter(ImageFilter.GaussianBlur(24))
    scr = ImageChops.add(scr, Image.merge("RGB", (fl, fl, fl)))

    vg = Image.new("L", (SW, SH), 0)
    vp = vg.load()
    cx, cy = SW / 2, SH / 2
    mx = math.hypot(cx, cy)
    for y in range(SH):
        for x in range(SW):
            vp[x, y] = int(9 * (math.hypot(x - cx, y - cy) / mx) ** 2.4)
    scr = ImageChops.subtract(scr, Image.merge("RGB", (vg, vg, vg)))

    # Textura de partícula do e-ink + a difusão da superfície.
    px = scr.load()
    random.seed(7)
    for y in range(SH):
        for x in range(SW):
            n = random.gauss(0, 1.3)
            r0, g0, b0 = px[x, y]
            px[x, y] = (max(0, min(255, int(r0 + n))),
                        max(0, min(255, int(g0 + n))),
                        max(0, min(255, int(b0 + n))))
    scr = scr.filter(ImageFilter.GaussianBlur(0.35))

    # Sombra que a moldura projeta sobre o vidro.
    sh = Image.new("L", (SW, SH), 0)
    ImageDraw.Draw(sh).rectangle([0, 0, SW - 1, SH - 1], outline=255, width=7)
    sh = sh.filter(ImageFilter.GaussianBlur(6))
    scr = Image.composite(Image.new("RGB", (SW, SH), (14, 14, 16)), scr,
                          sh.point(lambda v: int(v * 0.30)))

    # Reflexo de vidro, bem discreto.
    sn = Image.new("L", (SW, SH), 0)
    ImageDraw.Draw(sn).polygon([(-40, 0), (SW * 0.50, 0), (-40, SH * 0.58)], fill=255)
    sn = sn.filter(ImageFilter.GaussianBlur(80)).point(lambda v: int(v * 0.05))
    return Image.composite(Image.new("RGB", (SW, SH), (255, 255, 255)), scr, sn)


def rim(strength, lo=40):
    m = rrect(BW, BH, RAD, outline=True, width=2).filter(ImageFilter.GaussianBlur(1.1))
    m = ImageChops.multiply(m, vgrad(BW, BH, (255, 255, 255), (lo, lo, lo), 0.8).convert("L"))
    return m.point(lambda v: int(v * strength))


def front(scr):
    dev = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    dev.paste(vgrad(BW, BH, (48, 48, 51), (20, 20, 23), 1.15), (0, 0), rrect(BW, BH, RAD))
    dev.paste(scr, (SIDE, TOP))
    dev.paste(Image.new("RGB", (BW, BH), (152, 152, 160)), (0, 0), rim(0.62))
    d = ImageDraw.Draw(dev)
    d.rectangle([SIDE - 1, TOP - 1, SIDE + SW, TOP + SH], outline=(9, 9, 11, 255), width=1)
    tracked(d, BW / 2, TOP + SH + (BH - TOP - SH) / 2 - 24, "kindle",
            ImageFont.truetype(FONT, 34), (116, 116, 122, 255), 3.4)
    dev.putalpha(rrect(BW, BH, RAD))
    return dev


def back():
    dev = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    dev.paste(vgrad(BW, BH, (40, 40, 43), (17, 17, 20), 1.25), (0, 0), rrect(BW, BH, RAD))

    # Arco do chanfro da carcaça traseira.
    ss = 3
    a = Image.new("L", (BW * ss, BH * ss), 0)
    ImageDraw.Draw(a).arc([int(BW * 0.05 * ss), int(BH * 0.02 * ss),
                           int(BW * 1.28 * ss), int(BH * 0.40 * ss)],
                          start=8, end=104, fill=255, width=int(9 * ss))
    a = a.resize((BW, BH), Image.LANCZOS).filter(ImageFilter.GaussianBlur(7))
    dev.paste(Image.new("RGB", (BW, BH), (140, 140, 148)), (0, 0), a.point(lambda v: int(v * 0.42)))

    dg = Image.new("L", (BW, BH), 0)
    ImageDraw.Draw(dg).polygon([(0, 0), (BW * 0.85, 0), (0, BH * 0.75)], fill=255)
    dg = dg.filter(ImageFilter.GaussianBlur(110)).point(lambda v: int(v * 0.10))
    dev.paste(Image.new("RGB", (BW, BH), (190, 190, 200)), (0, 0), dg)

    dev.paste(Image.new("RGB", (BW, BH), (132, 132, 140)), (0, 0), rim(0.50, lo=30))
    dev.putalpha(rrect(BW, BH, RAD))
    return dev


def compose(cover_path, out_path, size=1280):
    cover = Image.open(cover_path).convert("RGB")
    scr = eink(cover)

    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    fx, fy = 0, 0
    bx, by = -int(BW * 0.50), -int(BH * 0.10)
    minx, miny = min(fx, bx), min(fy, by)
    maxx, maxy = max(fx, bx) + BW, max(fy, by) + BH
    ox = (size - (maxx - minx)) // 2 - minx
    oy = (size - (maxy - miny)) // 2 - miny
    FX, FY, BX, BY = fx + ox, fy + oy, bx + ox, by + oy

    def shadow(pos, blur, dy, op):
        lay = Image.new("L", (size, size), 0)
        lay.paste(rrect(BW, BH, RAD), (pos[0], pos[1] + dy))
        lay = lay.filter(ImageFilter.GaussianBlur(blur)).point(lambda v: int(v * op))
        canvas.paste(Image.new("RGB", (size, size), (122, 120, 122)), (0, 0), lay)

    shadow((BX, BY), 34, 26, 0.28)
    bk = back()
    canvas.paste(bk, (BX, BY), bk)
    # O aparelho da frente projeta sombra sobre o de trás.
    shadow((FX, FY), 30, 22, 0.32)
    shadow((FX, FY), 11, 6, 0.20)
    fr = front(scr)
    canvas.paste(fr, (FX, FY), fr)

    canvas.save(out_path)
    return canvas


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "capa.jpg"
    dst = sys.argv[2] if len(sys.argv) > 2 else "a-inventariante-kindle.png"
    compose(src, dst)
    print("gravado:", dst)
