"""cs_esenler.map üretici: Esenler Otogarı orta binası ve metro istasyonu, rehine kurtarma.

Eksenler: +x doğu, +y kuzey. Avlu zemini z=0. Orta bina x -576..576, y -448..448 (zemin kat
0..208, 1. kat 224..416). Metro: salondaki yürüyen merdiven kuyusu (x -32..448) -> bilet holü
(z -288) -> turnikeler -> merdiven -> peron (z -480, x 1216..1408, kuzey-güney). İkinci çıkış:
bilet holünün kuzeyinden merdivenle avluya (x 600..696, y ~960).
T: orta bina içi (salon + 1. kat). CT: güney ana girişi (dışarıdan) ve metro peronu (sırayla
karışık). Rehineler: bekleme salonu, PTT, 1. kat büroları. Kurtarma: güney giriş + metro peronu.
"""
import math
import os
from common import *

# ================================================================ DÜNYA
X0, X1, Y0, Y1, Z0, Z1 = -1920, 1920, -1600, 1600, -704, 1152
T = 32
W(box(X0 - T, Y0 - T, Z1, X1 + T, Y1 + T, Z1 + T, "sky"),
  box(X0 - T, Y0 - T, Z0 - T, X1 + T, Y1 + T, Z0, "es_concrete"),
  box(X0 - T, Y0, Z0, X0, Y1, Z1, "sky"), box(X1, Y0, Z0, X1 + T, Y1, Z1, "sky"),
  box(X0 - T, Y0 - T, Z0, X1 + T, Y0, Z1, "sky"), box(X0 - T, Y1, Z0, X1 + T, Y1 + T, Z1, "sky"))

# zemin dolgusu (asfalt üstlü); metro boşlukları aşağıda oyulur
FILL = [box(X0, Y0, Z0, X1, Y1, 0, {"top": "es_asphalt", "side": "es_concrete", "bottom": "es_concrete"})]

# ---- metro boşlukları
WELL = (-32, -96, -288, 448, 96, 0)                  # yürüyen merdiven kuyusu
MEZZ = (448, -256, -288, 928, 256, -96)              # bilet holü
STAIR_P = (928, -96, -480, 1216, 96, -96)            # perona inen merdiven
PLAT = (1216, -800, -480, 1408, 800, -288)           # peron
TRACK = (1408, -1056, -510, 1568, 1056, -288)        # ray yatağı + tünel (peron -478: 32 aşağı, geri atlanabilir)
EXIT_C = (600, 256, -288, 696, 512, -96)             # kuzey çıkış koridoru
EXIT_S = (600, 512, -288, 696, 944, 0)               # kuzey çıkış merdiveni
for hole in (WELL, MEZZ, STAIR_P, PLAT, TRACK, EXIT_C, EXIT_S):
    FILL = carve(FILL, hole)
W(FILL)

# ---- bilet holü, peron, tünel yüzeyleri (boşluk duvarlarına kaplama: ince plakalar)
MT = {"side": "es_mtile", "top": "es_ceil", "bottom": "es_granite"}
# bilet holü: zemin granit, tavan paneli, duvar fayans (boşluğun iç yüzlerine 4'lük kaplama)
def line_box(x0, y0, z0, x1, y1, z1, tex):
    return box(x0, y0, z0, x1, y1, z1, tex, align="local", fit=None)

W(box(448, -256, -288, 928, 256, -286, {"top": "es_granite", "side": "es_granite", "bottom": "es_granite"}))
W(box(448, -256, -96 - 2, 928, 256, -96, {"bottom": "es_ceil", "side": "es_ceil", "top": "es_ceil"}))
for (x0, y0, x1, y1) in ((448, -256, 928, -252), (448, 252, 928, 256), (924, -256, 928, -96), (924, 96, 928, 256), (448, -256, 452, -96), (448, 96, 452, 256)):
    W(box(x0, y0, -286, x1, y1, -98, "es_mtile", align="local"))
# peron: zemin, sarı çizgi, tavan, batı duvarı fayans; tünel duvarları isli
W(box(1216, -800, -480, 1408, 800, -478, {"top": "es_granite", "side": "es_granite", "bottom": "es_granite"}),
  box(1392, -800, -478, 1408, 800, -476, "es_yellow"),
  box(1216, -800, -290, 1408, 800, -288, {"bottom": "es_ceil", "side": "es_ceil", "top": "es_ceil"}),
  box(1216, -800, -478, 1220, -96, -290, "es_mtile", align="local"), box(1216, 96, -478, 1220, 800, -290, "es_mtile", align="local"),
  box(1220, -800, -478, 1408, -796, -290, "es_mtile", align="local"), box(1220, 796, -478, 1408, 800, -290, "es_mtile", align="local"),
  box(1408, -1056, -510, 1568, -1052, -288, "es_tunnel", align="local"), box(1408, 1052, -510, 1568, 1056, -288, "es_tunnel", align="local"),
  box(1564, -1056, -510, 1568, 1056, -288, "es_tunnel", align="local"),
  box(1408, -1056, -510, 1568, 1056, -508, {"top": "es_ballast", "side": "es_ballast", "bottom": "es_ballast"}))
for ry in (-1056, 1056):
    pass
for rx in (1440, 1520):
    W(box(rx, -1056, -508, rx + 6, 1056, -500, "es_rail"))
# tünel ağızlarında peron boşluğu dışında kalan tünel: karanlık; peron kenarı ile ray arası boşluk zaten var
# metro treni: 3 vagon (ray üstünde, y -600..600)
for k in range(3):
    y0 = -600 + k * 400 + (8 if k else 0)
    y1 = y0 + 392 - (8 if k == 2 else 0)
    W(box(1424, y0, -492, 1552, y1, -340, {"-x": "es_train", "+x": "es_train", "-y": "es_train_f", "+y": "es_train_f", "top": "es_bus_roof", "bottom": "es_metal"}, align="local", fit="uv"))
    group("tren")
    for wy in (y0 + 60, y1 - 60):
        D(box(1428, wy - 24, -508, 1548, wy + 24, -492, "es_metal"))
# turnikeler: x 752..768, 6 direk, aralar 40
group("turnike", shadow=True)
for i in range(7):
    yc = -240 + i * 80
    D(box(752, yc - 20, -288, 768, yc + 20, -248, "es_turnst", align="local"))
    if i < 6:
        D(box(752, yc + 20, -262, 768, yc + 60 - 8, -256, "es_metal"))      # kol (atlanabilir, alçak)
D(box(752, -256, -288, 768, -240, -200, "es_metal"), box(752, 240, -288, 768, 256, -200, "es_metal"))
# bilet gişesi (bilet holü güney duvarı önünde)
group("gise", shadow=True)
D(box(500, -252, -288, 640, -200, -240, "es_counter", align="local"), box(500, -252, -240, 640, -204, -98, "es_glass"))
D(relief("+y", -252, 660, 788, -200, -168, "es_s_bilet", 6), relief("+y", -252, 820, 884, -220, -156, "es_mmap", 6))
D(relief("-y", 252, 560, 816, -140, -108, "es_s_metrook", 6))
# merdivenler: salon -> bilet holü (yürüyen merdiven rampaları), bilet holü -> peron, bilet holü -> kuzey çıkış
W(ramp("x", -32, 448, 32, 96, 0, -288, {"top": "es_escal", "side": "es_metal", "slope": "es_escal"}),
  ramp("x", -32, 448, -96, -32, 0, -288, {"top": "es_escal", "side": "es_metal", "slope": "es_escal"}))
W(stairs("x", -32, 448, -32, 32, 0, -288, {"top": "es_granite", "side": "es_concrete"}, 16))
# kuyu tavanı (eğik) ve korkuluklar
W(Brush([(-32, -96, 208), (-32, 96, 208), (448, -96, -96), (448, 96, -96), (-32, -96, 232), (-32, 96, 232), (448, -96, -72), (448, 96, -72)],
        {"bottom": "es_ceil", "side": "es_concrete", "top": "es_concrete", "slope": "es_ceil"}))
group("korkuluk")
for yy_ in (-96, -32, 32, 96):
    D(Brush([(-32, yy_ - 3, 0), (-32, yy_ + 3, 0), (448, yy_ - 3, -288), (448, yy_ + 3, -288),
             (-32, yy_ - 3, 40), (-32, yy_ + 3, 40), (448, yy_ - 3, -248), (448, yy_ + 3, -248)], "es_metal"))
W(stairs("x", 928, 1216, -92, 92, -288, -480, {"top": "es_granite", "side": "es_concrete"}, 16))
W(box(928, -96, -480, 1216, -92, -96, "es_mtile", align="local"), box(928, 92, -480, 1216, 96, -96, "es_mtile", align="local"),
  box(1212, -92, -288, 1216, 92, -96, "es_mtile", align="local"),
  box(928, -92, -98, 1216, 92, -96, {"bottom": "es_ceil", "side": "es_ceil"}))
W(stairs("y", 512, 944, 604, 692, -288, 0, {"top": "es_granite", "side": "es_concrete"}, 16))
W(box(600, 256, -288, 604, 944, 0, "es_wall", align="local"), box(692, 256, -288, 696, 944, 0, "es_wall", align="local"),
  box(604, 256, -288, 692, 260, -96, "es_wall", align="local"))
W(box(600, 256, -288, 696, 512, -286, {"top": "es_granite", "side": "es_granite"}),
  box(600, 256, -98, 696, 512, -96, {"bottom": "es_ceil", "side": "es_ceil"}))
# kuzey çıkış barakası (yüzey): x 584..712, y 928..1024, giriş batıya
W(box(584, 928, 0, 600, 1024, 176, "es_fac_low"), box(696, 928, 0, 712, 1024, 176, "es_fac_low"),
  box(584, 1008, 0, 712, 1024, 176, "es_fac_low"), box(584, 928, 176, 712, 1024, 192, {"top": "es_bus_roof", "side": "es_metal", "bottom": "es_ceil"}))
group("tabela", shadow=True)
D(relief("-y", 928, 616, 680, 100, 164, "es_s_metro", 6), relief("+x", 712, 940, 1004, 100, 132, "es_s_cikis", 6))
# bilet holü kuzey duvar açıklığı (koridor zaten boşluk), peron tabelaları
group("tabela", shadow=True)
D(relief("-x", 1408, -200, 56, -400, -368, "es_s_metro2", 6), relief("-x", 1408, 300, 556, -400, -368, "es_s_metro2", 6))
D(relief("+x", 1220, -660, -596, -420, -356, "es_mmap", 6), relief("+x", 1220, 560, 624, -420, -356, "es_mmap", 6))
D(relief("+x", 1220, -300, -172, -440, -200, "es_sc", 6))                                   # peron afişi: sicakcikolata.com
# metro aydınlatması
for x in (520, 640, 760, 880):
    fluorescent(x - 32, -8, x + 32, 8, -98, brightness=110)
for y in range(-700, 701, 200):
    fluorescent(1280, y - 32, 1344, y + 32, -290, brightness=110, flicker=(y == -700))
light((1500, -900, -400), (200, 200, 230), 60); light((1500, 900, -400), (200, 200, 230), 60)
for x in (0, 150, 300):
    light((x, 0, -40 - x * 0.6), (240, 240, 255), 90)
light((1070, 0, -200), (240, 240, 255), 110); light((648, 400, -150), (240, 240, 255), 100); light((648, 700, -100), (240, 240, 255), 100)

# ================================================================ ÇEVRE BİNASI (yazıhaneler)
FAC_IN = {"s": -1344, "n": 1344, "w": -1664, "e": 1664}
GF, F1, ROOF = 176, 224, 760
W(box(X0, Y0, 0, X1, FAC_IN["s"], ROOF, {"+y": "es_facade", "side": "es_fac_low", "tb": "es_concrete"}, align="local"),
  box(X0, FAC_IN["n"], 0, X1, Y1, ROOF, {"-y": "es_facade", "side": "es_fac_low", "tb": "es_concrete"}, align="local"),
  box(X0, FAC_IN["s"], 0, FAC_IN["w"], FAC_IN["n"], ROOF, {"+x": "es_facade", "side": "es_fac_low", "tb": "es_concrete"}, align="local"),
  box(FAC_IN["e"], FAC_IN["s"], 0, X1, FAC_IN["n"], ROOF, {"-x": "es_facade", "side": "es_fac_low", "tb": "es_concrete"}, align="local"))
# zemin kat bandı: beton (yazıhane cepheleri kabartma)
group("cephe", shadow=True)
D(box(FAC_IN["w"], FAC_IN["s"], 0, -224, FAC_IN["s"] + 4, GF + 48, "es_fac_low", align="local"),
  box(224, FAC_IN["s"], 0, FAC_IN["e"], FAC_IN["s"] + 4, GF + 48, "es_fac_low", align="local"),
  box(-224, FAC_IN["s"], 176, 224, FAC_IN["s"] + 4, GF + 48, "es_fac_low", align="local"),
  box(FAC_IN["w"], FAC_IN["n"] - 4, 0, FAC_IN["e"], FAC_IN["n"], GF + 48, "es_fac_low", align="local"),
  box(FAC_IN["w"], FAC_IN["s"], 0, FAC_IN["w"] + 4, FAC_IN["n"], GF + 48, "es_fac_low", align="local"),
  box(FAC_IN["e"] - 4, FAC_IN["s"], 0, FAC_IN["e"], FAC_IN["n"], GF + 48, "es_fac_low", align="local"))
# CT ana giriş koridoru (güney cephenin ortası, dışarıdan): x -224..224, y -1600..-1344
W(box(-256, Y0, 0, -224, FAC_IN["s"] + 4, ROOF, "es_fac_low"), box(224, Y0, 0, 256, FAC_IN["s"] + 4, ROOF, "es_fac_low"))
ENTR = [box(-224, Y0, 0, 224, FAC_IN["s"] + 4, 176, "es_fac_low")]            # koridor boşluğu: facade bloğundan oyma yerine ayrı boşluk
# (çevre bloğunu oymak yerine koridoru boşluk olarak bırakıyoruz: cepheyi iki parçaya bölelim)
world[:] = [b for b in world if not (bounds(b)[0][1] == Y0 and bounds(b)[1][1] == FAC_IN["s"] and bounds(b)[0][0] == X0)]
W(box(X0, Y0, 0, -224, FAC_IN["s"], ROOF, {"+y": "es_facade", "side": "es_fac_low", "tb": "es_concrete"}, align="local"),
  box(224, Y0, 0, X1, FAC_IN["s"], ROOF, {"+y": "es_facade", "side": "es_fac_low", "tb": "es_concrete"}, align="local"),
  box(-224, Y0, 176, 224, FAC_IN["s"], ROOF, {"+y": "es_facade", "side": "es_fac_low", "tb": "es_concrete"}, align="local"),
  box(-224, Y0, 0, 224, Y0 + 16, 176, {"+y": "es_door_gl", "side": "es_fac_low"}, align="local", fit="uv"))
W(box(-224, Y0 + 16, 0, 224, FAC_IN["s"] + 4, 2, {"top": "es_terrazzo", "side": "es_concrete"}))
group("tabela", shadow=True)
D(relief("+y", FAC_IN["s"], -256, 256, 184, 248, "es_s_otogar", 6), relief("+y", FAC_IN["s"] - 0, -96, 32, 100, 132, "es_s_peron", 6) if False else None)
D(relief("+y", Y0 + 16, -200, 56, 130, 162, "es_s_cikis", 6))
light((0, -1480, 150), (255, 250, 240), 140); light((0, -1400, 150), (255, 250, 240), 120)

# yazıhane cepheleri + firma tabelaları + peron numaraları (güney ve kuzey), doğu/batı
COMP = ["es_c1", "es_c2", "es_c3", "es_c4", "es_c5", "es_c6"]
PNUM = ["es_p12", "es_p13", "es_p14", "es_p15", "es_p16", "es_p17", "es_p18", "es_p19"]
group("yazihane", shadow=True)
k = 0
for i, xc in enumerate((-1408, -1152, -896, -640, 640, 896, 1152, 1408)):
    for face, c in (("+y", FAC_IN["s"] + 4), ("-y", FAC_IN["n"] - 4)):
        D(relief(face, c, xc - 128, xc + 128, 0, 128, "es_office", 6))
        D(relief(face, c, xc - 128, xc + 128, 136, 200, COMP[(i + (0 if face == "+y" else 3)) % 6], 6))
    k += 1
for yc in (-896, -512, 512, 896):
    for face, c in (("+x", FAC_IN["w"] + 4), ("-x", FAC_IN["e"] - 4)):
        D(relief(face, c, yc - 128, yc + 128, 0, 128, "es_office", 6))
        D(relief(face, c, yc - 128, yc + 128, 136, 200, COMP[(yc // 256 + (0 if face == "+x" else 2)) % 6], 6))
# kaldırımlar (peronlar) + sundurma
SW = 160
W(box(FAC_IN["w"], FAC_IN["s"] + 4, 0, FAC_IN["e"], FAC_IN["s"] + SW, 16, {"top": "es_concrete", "side": "es_curb"}),
  box(FAC_IN["w"], FAC_IN["n"] - SW, 0, FAC_IN["e"], FAC_IN["n"] - 4, 16, {"top": "es_concrete", "side": "es_curb"}),
  box(FAC_IN["w"] + 4, FAC_IN["s"] + SW, 0, FAC_IN["w"] + SW, FAC_IN["n"] - SW, 16, {"top": "es_concrete", "side": "es_curb"}),
  box(FAC_IN["e"] - SW, FAC_IN["s"] + SW, 0, FAC_IN["e"] - 4, FAC_IN["n"] - SW, 16, {"top": "es_concrete", "side": "es_curb"}))
group("sundurma", shadow=True)
for xc in range(-1536, 1537, 256):
    if abs(xc) < 300:
        continue
    for yc, s in ((FAC_IN["s"] + SW - 24, 1), (FAC_IN["n"] - SW + 24, -1)):
        D(box(xc - 8, yc - 8, 16, xc + 8, yc + 8, 200, "es_pillar", align="local"))
for yc in range(-1024, 1025, 256):
    for xc in (FAC_IN["w"] + SW - 24, FAC_IN["e"] - SW + 24):
        D(box(xc - 8, yc - 8, 16, xc + 8, yc + 8, 200, "es_pillar", align="local"))
D(box(FAC_IN["w"], FAC_IN["s"] + 4, 200, -300, FAC_IN["s"] + SW, 212, {"top": "es_bus_roof", "bottom": "es_ceil", "side": "es_metal"}),
  box(300, FAC_IN["s"] + 4, 200, FAC_IN["e"], FAC_IN["s"] + SW, 212, {"top": "es_bus_roof", "bottom": "es_ceil", "side": "es_metal"}),
  box(FAC_IN["w"], FAC_IN["n"] - SW, 200, FAC_IN["e"], FAC_IN["n"] - 4, 212, {"top": "es_bus_roof", "bottom": "es_ceil", "side": "es_metal"}),
  box(FAC_IN["w"] + 4, FAC_IN["s"] + SW, 200, FAC_IN["w"] + SW, FAC_IN["n"] - SW, 212, {"top": "es_bus_roof", "bottom": "es_ceil", "side": "es_metal"}),
  box(FAC_IN["e"] - SW, FAC_IN["s"] + SW, 200, FAC_IN["e"] - 4, FAC_IN["n"] - SW, 212, {"top": "es_bus_roof", "bottom": "es_ceil", "side": "es_metal"}))
group("tabela", shadow=True)
for i, xc in enumerate((-1408, -1152, -896, -640, 640, 896, 1152, 1408)):
    D(relief("+y", FAC_IN["s"] + SW, xc - 64, xc + 64, 168, 200, PNUM[i], 4))
    D(relief("-y", FAC_IN["n"] - SW, xc - 64, xc + 64, 168, 200, PNUM[(i + 4) % 8], 4))
# sundurma altı lambalar
for xc in range(-1408, 1409, 512):
    for yc in (FAC_IN["s"] + 80, FAC_IN["n"] - 80):
        light((xc, yc, 190), (255, 244, 220), 90)


# ================================================================ OTOBÜSLER
def bus(xc, y_nose, direction, tex, bay_side=None, dest="ANKARA"):
    """y yönünde park etmiş otobüs. direction=-1: burnu güneyde (y_nose küçük), +1: burnu kuzeyde."""
    L, Wd = 560, 176
    if direction < 0:
        y0, y1 = y_nose, y_nose + L
        faces = {"-y": "es_bus_front", "+y": "es_bus_back"}
    else:
        y0, y1 = y_nose - L, y_nose
        faces = {"+y": "es_bus_front", "-y": "es_bus_back"}
    faces.update({"-x": tex, "+x": tex, "top": "es_bus_roof", "bottom": "es_metal"})
    W(box(xc - Wd / 2, y0, 24, xc + Wd / 2, y1, 208, faces, align="local", fit="uv"))
    group("otobus", shadow=True)
    for wy in (y0 + 92, y0 + 459) if direction < 0 else (y1 - 92, y1 - 459):
        for sx in (-1, 1):
            xe = xc + sx * Wd / 2
            D(box(min(xe, xe + sx * 8) - (24 if sx > 0 else 0), wy - 28, 0, max(xe, xe + sx * 8) + (24 if sx < 0 else 0), wy + 28, 56,
                  {"-x": "es_wheel", "+x": "es_wheel", "side": "es_metal", "tb": "es_metal"}, align="local", fit="uv"))
    if bay_side:
        xe = xc + bay_side * Wd / 2
        ym = (y0 + y1) / 2
        D(relief("+x" if bay_side > 0 else "-x", xe, ym - 64, ym + 64, 32, 96, "es_bay", 6))
        # bavullar
        bx = xe + bay_side * 40
        D(box(bx - 16, ym - 10, 0, bx + 16, ym + 22, 24, "es_bag1", align="local"), box(bx - 14, ym + 30, 0, bx + 14, ym + 52, 28, "es_bag2", align="local"),
          box(bx + bay_side * 30 - 12, ym - 40, 0, bx + bay_side * 30 + 12, ym - 16, 20, "es_bag3", align="local"))


def bus_x(yc, x_nose, direction, tex):
    """x yönünde park etmiş otobüs (doğu/batı cepheleri)."""
    L, Wd = 560, 176
    if direction < 0:
        x0, x1 = x_nose, x_nose + L
        faces = {"-x": "es_bus_front", "+x": "es_bus_back"}
    else:
        x0, x1 = x_nose - L, x_nose
        faces = {"+x": "es_bus_front", "-x": "es_bus_back"}
    faces.update({"-y": tex, "+y": tex, "top": "es_bus_roof", "bottom": "es_metal"})
    W(box(x0, yc - Wd / 2, 24, x1, yc + Wd / 2, 208, faces, align="local", fit="uv"))
    group("otobus", shadow=True)
    for wx in (x0 + 92, x0 + 459) if direction < 0 else (x1 - 92, x1 - 459):
        for sy in (-1, 1):
            ye = yc + sy * Wd / 2
            D(box(wx - 28, min(ye, ye + sy * 8) - (24 if sy > 0 else 0), 0, wx + 28, max(ye, ye + sy * 8) + (24 if sy < 0 else 0), 56,
                  {"-y": "es_wheel", "+y": "es_wheel", "side": "es_metal", "tb": "es_metal"}, align="local", fit="uv"))


S_NOSE, N_NOSE = FAC_IN["s"] + SW + 16, FAC_IN["n"] - SW - 16
bus(-1152, S_NOSE, -1, "es_bus1", bay_side=1)
bus(-832, S_NOSE, -1, "es_bus2")
bus(832, S_NOSE, -1, "es_bus3", bay_side=-1)
bus(1152, S_NOSE, -1, "es_bus1")
bus(-1152, N_NOSE, 1, "es_bus2")
bus(-832, N_NOSE, 1, "es_bus3", bay_side=1)
bus(832, N_NOSE, 1, "es_bus1")
bus(1152, N_NOSE, 1, "es_bus2", bay_side=-1)
bus_x(-320, FAC_IN["e"] - SW - 16, 1, "es_bus3")
bus_x(320, FAC_IN["e"] - SW - 16, 1, "es_bus1")
bus_x(-320, FAC_IN["w"] + SW + 16, -1, "es_bus2")
bus_x(320, FAC_IN["w"] + SW + 16, -1, "es_bus3")
# peron çizgileri (beyaz)
group("cizgi")
for xc in (-1312, -992, -672, 672, 992, 1312):
    D(box(xc - 4, FAC_IN["s"] + SW, 0, xc + 4, FAC_IN["s"] + SW + 600, 1, "es_white"), box(xc - 4, FAC_IN["n"] - SW - 600, 0, xc + 4, FAC_IN["n"] - SW, 1, "es_white"))

# ================================================================ ORTA BİNA
BX0, BY0, BX1, BY1 = -576, -448, 576, 448
EXT = {"side": "es_fac_low", "tb": "es_concrete"}
EXT1 = {"side": "es_facade", "tb": "es_concrete"}
# zemin kat dış duvarları: doğu/batı ana girişler (160), güney/kuzey kapılar (96)
W(wall_y(BY0, BY0 + 32, BX0, BX1, 0, 208, [(-48, 48, 0, 160)], EXT))
W(wall_y(BY1 - 32, BY1, BX0, BX1, 0, 208, [(-48, 48, 0, 160)], EXT))
W(wall_x(BX0, BX0 + 32, BY0 + 32, BY1 - 32, 0, 208, [(-80, 80, 0, 160)], EXT))
W(wall_x(BX1 - 32, BX1, BY0 + 32, BY1 - 32, 0, 208, [(-80, 80, 0, 160)], EXT))
# 1. kat döşemesi (merdiven boşlukları oyulur), dış duvarları (pencereli doku), çatı, parapet
SLAB = [box(BX0, BY0, 208, BX1, BY1, 224, {"top": "es_terrazzo", "bottom": "es_ceil", "side": "es_concrete"})]
SLAB = carve(SLAB, (-544, -160, 207, -448, 160, 225))
SLAB = carve(SLAB, (448, -160, 207, 544, 160, 225))
W(SLAB)
W(box(BX0, BY0, 224, BX1, BY0 + 32, 416, EXT1, align="local"), box(BX0, BY1 - 32, 224, BX1, BY1, 416, EXT1, align="local"),
  box(BX0, BY0 + 32, 224, BX0 + 32, BY1 - 32, 416, EXT1, align="local"), box(BX1 - 32, BY0 + 32, 224, BX1, BY1 - 32, 416, EXT1, align="local"))
W(box(BX0, BY0, 416, BX1, BY1, 432, {"top": "es_concrete", "bottom": "es_ceil", "side": "es_concrete"}),
  box(BX0 - 8, BY0 - 8, 432, BX1 + 8, BY0 + 24, 480, "es_concrete"), box(BX0 - 8, BY1 - 24, 432, BX1 + 8, BY1 + 8, 480, "es_concrete"),
  box(BX0 - 8, BY0 - 8, 432, BX0 + 24, BY1 + 8, 480, "es_concrete"), box(BX1 - 24, BY0 - 8, 432, BX1 + 8, BY1 + 8, 480, "es_concrete"))
# 1. kat kırılabilir pencereler (dış duvarda açıklık + cam)
for (x0, y0, x1, y1) in ((-480, BY0, -352, BY0 + 32), (352, BY0, 480, BY0 + 32), (-480, BY1 - 32, -352, BY1), (352, BY1 - 32, 480, BY1)):
    world[:] = [b for b in world if not (bounds(b)[0][0] == BX0 and bounds(b)[0][2] == 224 and bounds(b)[1][2] == 416 and (bounds(b)[0][1] == y0))] if False else world
# (pencereleri duvarlara oyarak ekle)
def cut_window(wall_sel, hole, glass_box):
    global world
    keep, cut = [], []
    for b in world:
        (bx0, by0, bz0), (bx1, by1, bz1) = bounds(b)
        if wall_sel(bx0, by0, bz0, bx1, by1, bz1):
            cut.append(b)
        else:
            keep.append(b)
    world[:] = keep + carve(cut, hole)
    ent("func_breakable", brushes=[glass_box], material=0, health=20, rendermode=2, renderamt=160)

for xw in (-416, 416):
    cut_window(lambda x0, y0, z0, x1, y1, z1: y0 == BY0 and y1 == BY0 + 32 and z0 == 224, (xw - 56, BY0 - 1, 288, xw + 56, BY0 + 33, 368),
               box(xw - 56, BY0 + 12, 288, xw + 56, BY0 + 16, 368, "es_glass"))
    cut_window(lambda x0, y0, z0, x1, y1, z1: y0 == BY1 - 32 and y1 == BY1 and z0 == 224, (xw - 56, BY1 - 33, 288, xw + 56, BY1 + 1, 368),
               box(xw - 56, BY1 - 16, 288, xw + 56, BY1 - 12, 368, "es_glass"))
for yw in (-280, 280):
    cut_window(lambda x0, y0, z0, x1, y1, z1: x0 == BX0 and x1 == BX0 + 32 and z0 == 224, (BX0 - 1, yw - 56, 288, BX0 + 33, yw + 56, 368),
               box(BX0 + 12, yw - 56, 288, BX0 + 16, yw + 56, 368, "es_glass"))
    cut_window(lambda x0, y0, z0, x1, y1, z1: x0 == BX1 - 32 and x1 == BX1 and z0 == 224, (BX1 - 33, yw - 56, 288, BX1 + 1, yw + 56, 368),
               box(BX1 - 16, yw - 56, 288, BX1 - 12, yw + 56, 368, "es_glass"))
# zemin kat döşemesi (terrazzo) ve tavan; kuyu oyulur
FLOOR = [box(BX0 + 32, BY0 + 32, 0, BX1 - 32, BY1 - 32, 2, {"top": "es_terrazzo", "side": "es_terrazzo", "bottom": "es_terrazzo"})]
FLOOR = carve(FLOOR, (WELL[0], WELL[1], -1, WELL[3], WELL[4], 3))
W(FLOOR)
# iç bölmeler (zemin): PTT (kuzeybatı), Polis (kuzeydoğu), bekleme (güneybatı), büfe (güneydoğu)
IW = {"side": "es_wall", "tb": "es_wall"}
W(wall_y(160, 176, -544, -224, 0, 208, [(-424, -344, 0, 128)], IW), box(-240, 176, 0, -224, 416, 208, IW))        # PTT
W(wall_y(160, 176, 224, 544, 0, 208, [(344, 424, 0, 128)], IW), box(224, 176, 0, 240, 416, 208, IW))             # Polis
W(wall_y(-176, -160, -544, -96, 0, 208, [(-400, -240, 0, 160)], IW), box(-112, -416, 0, -96, -176, 208, IW))      # bekleme salonu
W(wall_y(-176, -160, 224, 544, 0, 208, [(300, 460, 0, 160)], IW), box(224, -416, 0, 240, -176, 208, IW))         # büfe
# merdivenler (batı ve doğu): y -160 -> 160 arası 208 yükselir
W(stairs("y", -160, 160, -544, -448, 0, 208, {"top": "es_terrazzo", "side": "es_concrete"}, 16))
W(stairs("y", 160, -160, 448, 544, 0, 208, {"top": "es_terrazzo", "side": "es_concrete"}, 16))
group("korkuluk")
D(box(-448, -160, 208, -442, 160, 248, "es_metal"), box(442, -160, 208, 448, 160, 248, "es_metal"))
# 1. kat iç bölmeler: orta koridor y -96..96, odalar: NW (EMANET), NE (YÖNETİM), SW (BİLET), SE (BÜRO)
W(wall_y(96, 112, -448, 448, 224, 416, [(-300, -236, 224, 320), (236, 300, 224, 320)], IW), box(-8, 112, 224, 8, 416, 416, IW))
W(wall_y(-112, -96, -448, 448, 224, 416, [(-300, -236, 224, 320), (236, 300, 224, 320)], IW), box(-8, -416, 224, 8, -112, 416, IW))
# tavanlar: zemin kat tavanı = döşeme altı; 1. kat tavanı = çatı altı (es_ceil zaten). Floresanlar
for x in (-400, -200, 0, 200, 400):
    fluorescent(x - 48, -8, x + 48, 8, 208, brightness=140)
for (x, y) in ((-384, 300), (384, 300), (-320, -300), (384, -300)):
    fluorescent(x - 48, y - 8, x + 48, y + 8, 208, brightness=130, flicker=(x == 384 and y == -300))
for x in (-300, 0, 300):
    fluorescent(x - 48, -8, x + 48, 8, 416, brightness=130)
for (x, y) in ((-228, 264), (228, 264), (-228, -264), (228, -264)):
    fluorescent(x - 48, y - 8, x + 48, y + 8, 416, brightness=120)
# dış tabelalar, M işareti, PTT/POLİS, bekleme salonu eşyaları
group("tabela", shadow=True)
D(relief("-y", BY0, -256, 256, 300, 364, "es_s_otogar", 8), relief("+y", BY1, -256, 256, 300, 364, "es_s_otogar", 8))
D(relief("-x", BX0, -32, 32, 180, 244, "es_s_metro", 8), relief("+x", BX1, -32, 32, 180, 244, "es_s_metro", 8))
D(relief("-y", 160, -416, -352, 136, 168, "es_s_ptt", 6), relief("-y", 160, 320, 448, 136, 168, "es_s_polis", 6))
D(relief("+y", -160, -400, -144, 170, 202, "es_s_metrook", 6) if False else None)
D(relief("+y", -160, 300, 428, 170, 202, "es_s_bufe", 6), relief("-y", 112, -392, -264, 330, 362, "es_s_emanet", 6), relief("+y", -96, 236, 364, 330, 362, "es_s_bilet", 6))
D(relief("+x", -32, -200, -72, 60, 300, "es_sc", 6) if False else None)
D(relief("-x", -544 + 4, -60, 60, 60, 92, "es_s_metrook", 6) if False else None)
# bekleme salonu: koltuk sıraları; PTT gişesi; büfe; polis masası
group("esya", shadow=True)
for yr in (-360, -300, -240):
    D(box(-520, yr - 12, 0, -200, yr + 12, 24, "es_seat", align="local"), box(-520, yr + 12, 24, -200, yr + 20, 56, "es_seat", align="local"))
D(box(-540, 184, 0, -300, 232, 44, "es_counter", align="local"), box(-540, 184, 44, -300, 188, 140, "es_glass"))
D(box(300, -412, 0, 540, -372, 44, "es_counter", align="local"), relief("+y", -372, 320, 448, 44, 172, "es_kiosk", 8))
D(box(300, 300, 0, 440, 340, 44, "es_counter", align="local"))
D(box(-40, 160, 0, 40, 176, 2, "es_sc", align="local") if False else None)
D(relief("+y", 176, -500, -436, 60, 188, "es_poster1", 6), relief("+y", 176, 380, 444, 60, 188, "es_poster2", 6), relief("-y", -176, 380, 444, 60, 188, "es_poster1", 6))
D(relief("-x", -224, 230, 294, 40, 160, "es_sc", 6))                                      # sicakcikolata.com afişi (PTT duvarı)
# 1. kat eşyaları
D(box(-430, 150, 224, -300, 190, 268, "es_counter", align="local"), box(300, 150, 224, 430, 190, 268, "es_counter", align="local"),
  box(-430, -230, 224, -300, -190, 268, "es_counter", align="local"), box(300, -230, 224, 430, -190, 268, "es_counter", align="local"))
for x in (-380, 380):
    D(box(x - 64, 250, 224, x + 64, 300, 290, "es_shelf", align="local", fit="uv"))
# dış: ana giriş camları (yan paneller), kapı üstü sundurma
group("cam")
for sx, xx_ in ((-1, BX0), (1, BX1)):
    D(box(xx_ + (12 if sx < 0 else -16), -176, 0, xx_ + (16 if sx < 0 else -12), -80, 160, "es_glass"), box(xx_ + (12 if sx < 0 else -16), 80, 0, xx_ + (16 if sx < 0 else -12), 176, 160, "es_glass"))
    D(box(xx_ - 96 * (1 if sx < 0 else 0) - (0 if sx < 0 else 0) + (0 if sx < 0 else 0), -120, 168, xx_ + (0 if sx < 0 else 96), 120, 180, {"top": "es_bus_roof", "bottom": "es_ceil", "side": "es_metal"}) if False else None)
for sx in (-1, 1):
    x0_, x1_ = (BX0 - 96, BX0) if sx < 0 else (BX1, BX1 + 96)
    D(box(x0_, -120, 168, x1_, 120, 180, {"top": "es_bus_roof", "bottom": "es_ceil", "side": "es_metal"}))
    light((BX0 - 48 if sx < 0 else BX1 + 48, 0, 160), (255, 244, 220), 100)

# ================================================================ FİGÜRLER (otogar insanları)
# yazıhaneciler (yolcu çeken): yazıhane kapıları önü, kaldırımda
figure(-1050, FAC_IN["s"] + 60, 16, 90, 1)
figure(-560, FAC_IN["s"] + 60, 16, 90, 3, blink=(14, 7))
figure(700, FAC_IN["s"] + 60, 16, 90, 0, blink=(11, 9))
figure(950, FAC_IN["n"] - 60, 16, 270, 2)
figure(-700, FAC_IN["n"] - 60, 16, 270, 1, blink=(16, 6))
figure(FAC_IN["w"] + 60, -700, 16, 0, 3)
# muavinler (bagaj yerleştiren): açık bagaj bölmelerinin önünde
figure(-1152 + 88 + 44, S_NOSE + 280 + 40, 0, 180, 2, blink=(12, 8))
figure(832 - 88 - 44, S_NOSE + 280 - 40, 0, 0, 0)
figure(-832 + 88 + 44, N_NOSE - 280 + 40, 0, 180, 1, blink=(9, 11))
figure(1152 - 88 - 44, N_NOSE - 280 - 40, 0, 0, 3)
# ikramcılar (çay tepsisi): büfe önü ve bekleme salonu; batı kaldırımı
figure(380, -330, 2, 90, 1)
group("tepsi"); D(box(368, -304, 40, 392, -296, 42, "es_metal"), box(372, -303, 42, 378, -298, 48, "es_tea", align="local"), box(382, -303, 42, 388, -298, 48, "es_tea", align="local"))
figure(-300, -200, 2, 270, 2, blink=(10, 10))
figure(FAC_IN["w"] + 60, 300, 16, 0, 0, blink=(13, 9))
# bekleyen yolcular (salon)
figure(-470, -330, 2, 0, 3)
figure(-250, -270, 2, 90, 0)
figure(820, -1000, 0, 90, 2, blink=(15, 15))          # peron arasında yolcu

# ================================================================ OYUN VARLIKLARI
# CT: güney giriş koridoru (dışarıdan) ve metro peronu; sırayla karışık
for i in range(8):
    x = -168 + (i % 4) * 112
    y = -1560 + (i // 4) * 64
    ent("info_player_start", (x, y, 40), angles="0 90 0")
    ent("info_player_start", (1312 - (i % 2) * 48, -600 + (i // 2) * 120, -440), angles="0 180 0")
# T: salon (zemin) + 1. kat
for i in range(8):
    ent("info_player_deathmatch", (-400 + (i % 4) * 100, -60 + (i // 4) * 120, 40), angles="0 0 0")
for i in range(8):
    ent("info_player_deathmatch", (-300 + (i % 4) * 150, -40 + (i // 4) * 80, 264), angles="0 0 0")
# rehineler
for (x, y, z, a) in ((-300, -200, 2, 90), (-400, 300, 2, 270), (300, 340, 224, 270), (-380, -300, 224, 90)):
    ent("hostage_entity", (x, y, z), angles="0 %d 0" % a)
ent("func_hostage_rescue", brushes=[box(-224, Y0 + 16, 0, 224, FAC_IN["s"], 176, "aaatrigger")])
ent("func_hostage_rescue", brushes=[box(1216, -800, -480, 1408, 800, -300, "aaatrigger")])
ent("info_map_parameters", (0, 0, 600), buying=0)

# ================================================================ IŞIK ve SESLER
ent("light_environment", (0, 0, 1000), angles="0 250 0", pitch=-60, _light="255 248 236 230", _diffuse_light="190 205 230 160")
for (x, y) in ((-1024, -800), (1024, -800), (-1024, 800), (1024, 800), (0, -800), (0, 800), (-900, 0), (900, 0)):
    light((x, y, 400), (255, 250, 240), 60)                  # avlu dolgu ışığı
# sesler
for (x, y) in ((-992, -900), (992, -900), (-992, 900), (992, 900), (1400, 0), (-1400, 0)):
    sound((x, y, 100), "esenler/otobus.wav", 6, 4)
sound((0, 0, 150), "esenler/kalabalik.wav", 6, 4)
sound((-350, -300, 150), "esenler/kalabalik.wav", 5, 2)
sound((688, 0, -200), "esenler/floresan.wav", 4, 2)
sound((1312, 0, -400), "esenler/metro.wav", 6, 8)
sound((0, -900, 300), "esenler/ruzgar.wav", 4, 8)
sound((0, 900, 300), "esenler/ruzgar.wav", 4, 8)
sound((1312, 0, -400), "esenler/metro_gelis.wav", 8, 4 + 16 + 32, targetname="ses_tren")
sound((0, 0, 200), "esenler/anons.wav", 8, 8 + 16 + 32, targetname="ses_anons")
sound((-700, -1200, 100), "esenler/cigirtkan.wav", 8, 4 + 16 + 32, targetname="ses_cig1")
sound((900, 1200, 100), "esenler/cigirtkan2.wav", 8, 4 + 16 + 32, targetname="ses_cig2")
sound((-1000, -900, 60), "esenler/bagaj.wav", 7, 4 + 16 + 32, targetname="ses_bagaj")
sound((1000, 900, 60), "esenler/fren.wav", 7, 4 + 16 + 32, targetname="ses_fren")
ents.append(({"classname": "multi_manager", "targetname": "mm_sesler", "spawnflags": "1",
              "ses_anons": "6", "ses_cig1": "15", "ses_bagaj": "22", "ses_tren": "30", "ses_cig2": "41", "ses_fren": "50", "mm_sesler": "62"}, []))
ent("trigger_auto", (0, 0, 64), target="mm_sesler", spawnflags=1)

if __name__ == "__main__":
    write(os.path.join(HERE, "cs_esenler.map"), "esenler", "Esenler Otogari")
