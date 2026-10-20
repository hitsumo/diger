"""de_esenler.map üretici: Esenler Otogarı'nın alt katı, labirent gibi eski dükkânlar, bomba imha.

Her şey yer altında (z=0 ana kat, tavan 176); yalnızca T'nin servis avlusu (z=192) açık havada.
Koridorlar: güney C1 (y -512..-352), orta C2 (y -80..80), kuzey C3 (y 352..512); ara koridorlar
X1 (x -1056..-896), X2 (x -80..80), X3 (x 896..1056). Dükkânlar bloklara oyulmuş odalar;
kapalı olanlar kepenk kabartması. A: doğudaki tuvalet (zemin -32). B: batıdaki depo (zemin -64),
yanında yükleme rıhtımı ve servis rampasıyla T'nin avlusu. CT: kuzeydeki ana merdiven sahanlığı
(bozuk yürüyen merdivenler, üst kapı kapalı).
"""
import math
import os
from common import *

X0, X1, Y0, Y1, Z0, Z1 = -1728, -1344, 1728, 1344, -320, 704
X0, Y0, X1, Y1 = -1728, -1344, 1728, 1344
T = 32
W(box(X0 - T, Y0 - T, Z1, X1 + T, Y1 + T, Z1 + T, "sky"),
  box(X0 - T, Y0 - T, Z0 - T, X1 + T, Y0, Z1, "es_concrete"), box(X0 - T, Y1, Z0 - T, X1 + T, Y1 + T, Z1, "es_concrete"),
  box(X0 - T, Y0, Z0 - T, X0, Y1, Z1, "es_concrete"), box(X1, Y0, Z0 - T, X1 + T, Y1, Z1, "es_concrete"),
  box(X0, Y0, Z0 - T, X1, Y1, Z0, "es_concrete"))

FILL = [box(X0, Y0, Z0, X1, Y1, Z1, {"top": "es_tile_brn", "bottom": "es_ceil", "side": "es_wall"})]
VOIDS = {
    "C1": (-1600, -512, 0, 1200, -352, 176), "C2": (-1056, -80, 0, 1056, 80, 176), "C3": (-1056, 352, 0, 1056, 512, 176),
    "X1": (-1056, -512, 0, -896, 512, 176), "X2": (-80, -512, 0, 80, 512, 176), "X3": (896, -512, 0, 1056, 512, 176),
    "DOCK": (-1600, -512, 0, -1056, 80, 192), "RAMP": (-1300, -900, 0, -1100, -512, 368), "YARD": (-1600, -1344, 192, -1056, -900, 704),
    "DEPO": (-1600, 80, -64, -1056, 512, 176),
    "WC": (1200, -512, -32, 1600, -80, 160), "WC2": (1056, -176, 0, 1200, -96, 160),
    "LOK": (160, -320, 0, 816, -80, 160), "LOKK": (400, -352, 0, 480, -320, 160),
    "BERBER": (-600, -280, 0, -400, -80, 160), "CAY": (-300, -160, 0, -160, -80, 160), "S1": (-700, -352, 0, -520, -160, 160),
    "N1": (-400, 80, 0, -240, 352, 160), "EMANET": (-760, 80, 0, -560, 240, 160), "KUAFOR": (200, 192, 0, 400, 352, 160),
    "BUFE": (500, 80, 0, 700, 240, 160), "N2": (760, 80, 0, 880, 352, 160), "S2": (-700, 512, 0, -500, 672, 160),
    "CTH": (-200, 512, 0, 200, 900, 368), "CTL": (-200, 900, 192, 200, 1100, 368),
}
for v in VOIDS.values():
    FILL = carve(FILL, v)
W(FILL)


def plates(v, tex, faces=("-x", "+x", "-y", "+y"), z0=None, z1=None, t=4, holes=()):
    """Boşluğun iç yüzlerine kaplama plakaları; holes: (x0,y0,z0,x1,y1,z1) açıklıkları plakalardan oyulur."""
    x0, y0, vz0, x1, y1, vz1 = v
    z0 = vz0 if z0 is None else z0
    z1 = vz1 if z1 is None else z1
    out = []
    if "-x" in faces:
        out.append(box(x0, y0, z0, x0 + t, y1, z1, tex, align="local"))
    if "+x" in faces:
        out.append(box(x1 - t, y0, z0, x1, y1, z1, tex, align="local"))
    if "-y" in faces:
        out.append(box(x0, y0, z0, x1, y0 + t, z1, tex, align="local"))
    if "+y" in faces:
        out.append(box(x0, y1 - t, z0, x1, y1, z1, tex, align="local"))
    for h in holes:
        out = carve(out, h)
    return out


def floor_slab(v, tex, z=None):
    x0, y0, vz0, x1, y1, vz1 = v
    z = vz0 if z is None else z
    return box(x0, y0, z, x1, y1, z + 2, {"top": tex, "side": tex, "bottom": tex})


# ================================================================ ODALAR
# --- servis avlusu (T), rampa, rıhtım, depo (B)
W(floor_slab(VOIDS["YARD"], "es_asphalt"), plates(VOIDS["YARD"], "es_concrete", ("-x", "+x", "-y"), 192, 704))
W(box(-1056, -1344, 192, -1052, -900, 704, "es_concrete", align="local"))
W(ramp("y", -512, -900, -1300, -1100, 0, 192, {"top": "es_asphalt", "slope": "es_asphalt", "side": "es_concrete"}))
W(plates(VOIDS["RAMP"], "es_concrete", ("-x", "+x")), plates(VOIDS["DOCK"], "es_concrete", ("-x", "-y"), holes=[(-1301, -520, -1, -1099, -500, 200)]))
W(box(-1600, -904, 192, -1300, -900, 704, "es_concrete", align="local"), box(-1100, -904, 192, -1056, -900, 704, "es_concrete", align="local"),
  box(-1300, -904, 368, -1100, -900, 704, "es_concrete", align="local"))
W(floor_slab(VOIDS["DOCK"], "es_asphalt"))
W(box(-1600, -512, 0, -1056, -508, 192, "es_concrete", align="local") if False else None) if False else None
W(floor_slab(VOIDS["DEPO"], "es_concrete"), plates(VOIDS["DEPO"], "es_concrete", ("-x", "+y")))
# depo girişi: C3/X1 tarafından (x=-1056) rampa ile -64'e; rıhtımdan (y=80) kepenkli geniş açıklık
W(ramp("x", -1056, -1200, -1056 + 0, -1056, 0, 0, "es_concrete") if False else None) if False else None
W(ramp("x", -1056, -1216, 360, 504, 0, -64, {"top": "es_concrete", "slope": "es_concrete", "side": "es_concrete"}))
W(box(-1600, 80, -64, -1420, 84, 176, "es_concrete", align="local"), box(-1180, 80, -64, -1056, 84, 176, "es_concrete", align="local"),
  box(-1420, 80, 120, -1180, 84, 176, "es_concrete", align="local"))      # rıhtım-depo arası: açıklık x -1420..-1180, 120 yüksek
W(box(-1056, 80, 0, -1052, 360, 176, "es_wall", align="local"), box(-1056, 504, 0, -1052, 512, 176, "es_wall", align="local"))   # depo doğu duvarı (X1 tarafı); rampa y 360..504
# rıhtım: kamyon, paletler, variller
group("rihtim", shadow=True)
D(box(-1560, -480, 0, -1300, -300, 200, {"side": "es_rust", "tb": "es_metal"}), box(-1300, -480, 0, -1200, -300, 150, "es_metal"))   # kamyon kasası + kabin
D(box(-1400, -200, 0, -1200, -100, 40, "es_crate", align="local"), box(-1540, -200, 0, -1460, -120, 60, "es_sack", align="local"))
D(box(-1200, 0, 0, -1100, 60, 56, "es_boxes", align="local"), box(-1560, 0, 0, -1480, 70, 64, "es_boxes", align="local"))
D(relief("+y", -508, -1500, -1244, 60, 92, "es_s_girilmez", 6) if False else None)
# avlu: konteyner, variller; sahte otobüs tamponu
D(box(-1580, -1000, 192, -1340, -910, 320, {"side": "es_rust", "tb": "es_metal"}), box(-1150, -1320, 192, -1070, -1240, 250, "es_boxes", align="local"))
# depo: raflar, koliler, çuvallar; B tabelası
group("depo", shadow=True)
for y0 in (120, 260, 400):
    D(box(-1596, y0, -64, -1540, y0 + 96, 64, "es_shelf", align="local", fit="uv"))
D(box(-1400, 380, -64, -1300, 480, -8, "es_boxes", align="local"), box(-1300, 400, -64, -1240, 460, -20, "es_boxes", align="local"),
  box(-1500, 160, -64, -1420, 240, -16, "es_sack", align="local"), box(-1250, 140, -64, -1160, 230, -24, "es_crate", align="local"),
  box(-1160, 300, -64, -1100, 360, -30, "es_crate", align="local"))
D(relief("+x", -1596, 300, 428, 60, 92, "es_s_depo", 6) if False else None)
group("tabela", shadow=True)
D(relief("-x", -1056, 380, 508, 100, 132, "es_s_depo", 6))
light((-1300, 300, 120), (255, 230, 190), 220); light((-1500, 450, 120), (255, 230, 190), 120, style=10); light((-1150, 200, 120), (255, 230, 190), 150)
light((-1300, -300, 160), (255, 240, 220), 200); light((-1150, -100, 160), (255, 240, 220), 160); light((-1500, -60, 160), (255, 240, 220), 140)
light((-1200, -700, 300), (255, 240, 220), 200); light((-1200, -560, 150), (255, 240, 220), 150)
# avlu: duvar projektörleri (gündüz ama derin kuyu)
for (x, y) in ((-1560, -1000), (-1100, -1000), (-1560, -1300), (-1100, -1300)):
    light((x, y, 420), (255, 246, 230), 260)
light((-1330, -1120, 600), (255, 246, 230), 200)
# --- tuvalet (A): zemin -32, fayans duvar, kabinler, pisuarlar, lavabolar
W(floor_slab(VOIDS["WC"], "es_tile_wet"), plates(VOIDS["WC"], "es_wtile", ("-x", "+x", "-y", "+y"), holes=[(1199, -513, -33, 1205, -351, 161), (1199, -177, -33, 1205, -95, 161)]))
W(stairs("x", 1200, 1248, -512, -352, 0, -32, {"top": "es_tile_wet", "side": "es_wtile"}, 16))
W(stairs("x", 1200, 1248, -176, -96, 0, -32, {"top": "es_tile_wet", "side": "es_wtile"}, 16))
W(plates(VOIDS["WC2"], "es_wtile", ("-y", "+y")))
group("tuvalet", shadow=True)
for i in range(5):
    x0 = 1270 + i * 66
    D(box(x0, -176, -32, x0 + 4, -84, 128, "es_wtile", align="local"))                                  # kabin yan duvarı
    D(box(x0 + 8, -180, -24, x0 + 60, -176, 120, "es_door_wc", align="local", fit="uv") if i != 2 else None)   # kapı (biri açık/yok)
    D(box(x0 + 16, -150, -32, x0 + 52, -100, 0, "es_sink", align="local"))                             # alaturka/klozet (basit kütle)
D(box(1270 + 5 * 66, -176, -32, 1270 + 5 * 66 + 4, -84, 128, "es_wtile", align="local"))
for y in (-460, -400, -340, -280):
    D(relief("-x", 1600, y - 24, y + 24, 20, 84, "es_urinal", 8))
for x in (1300, 1380, 1460):
    D(box(x - 28, -512, 8, x + 28, -480, 24, "es_sink", align="local", fit="uv"), relief("+y", -512, x - 28, x + 28, 44, 108, "es_mirror", 4))
D(box(1216, -500, -32, 1240, -460, 40, "es_metal"), box(1560, -140, -32, 1596, -100, 30, "es_boxes", align="local"))   # temizlik kovası, karton
D(relief("-y", -80, 1300, 1364, 170, 202, "es_s_wc", 6) if False else None)
group("tabela", shadow=True)
D(relief("-y", -352, 1072, 1136, 136, 168, "es_s_wc", 6), relief("-y", -352, 1140, 1180, 100, 228, "es_poster1", 6))
D(relief("+x", 1200, -480, -400, 60, 300, "es_sc", 6) if False else None)
# ıslak zemin lekeleri (düşük plakalar) ve boru
D(box(1420, -300, -31, 1560, -200, -30, "es_tile_wet", align="local"), box(1230, -400, -31, 1330, -330, -30, "es_tile_wet", align="local"))
D(box(1596, -512, 100, 1600, -80, 116, "es_pipe", align="local") if False else None)
D(box(1200, -84, 120, 1600, -80, 136, "es_pipe", align="local"))
light((1300, -300, 120), (210, 240, 200), 120, style=10); light((1500, -200, 120), (210, 240, 200), 100); light((1400, -450, 120), (230, 240, 210), 110)
light((1128, -430, 140), (240, 240, 230), 100); light((1128, -136, 130), (240, 240, 230), 80)
sound((1400, -300, 60), "esenler/damla.wav", 7, 2)
# --- lokanta
W(floor_slab(VOIDS["LOK"], "es_terrazzo"), plates(VOIDS["LOK"], "es_wall_up", ("-x", "+x", "-y"), holes=[(399, -321, -1, 481, -315, 161)]))
W(box(160, -84, 0, 400, -80, 160, "es_wall", align="local"), box(560, -84, 0, 816, -80, 160, "es_wall", align="local"))   # C2'ye açıklık x 400..560
W(box(160, -84, 112, 816, -80, 160, "es_wall", align="local") if False else None) if False else None
group("lokanta", shadow=True)
D(box(620, -320, 0, 800, -260, 44, "es_counter", align="local"), box(620, -264, 44, 800, -260, 120, "es_glass"))      # tezgâh + vitrin
for (x, y) in ((260, -200), (260, -280), (420, -200), (420, -280), (560, -220)):
    D(box(x - 32, y - 32, 40, x + 32, y + 32, 44, "es_table", align="local"), box(x - 4, y - 4, 0, x + 4, y + 4, 40, "es_metal"))
    for dx, dy in ((-44, 0), (44, 0), (0, -44), (0, 44)):
        D(box(x + dx - 10, y + dy - 10, 0, x + dx + 10, y + dy + 10, 24, "es_seat", align="local"))
D(relief("-y", -84, 420, 540, 56, 88, "es_s_lokanta", 6) if False else None)
group("tabela", shadow=True)
D(relief("-y", -80, 352, 608, 118, 150, "es_s_lokanta", 6))
D(relief("-x", 816, -300, -236, 40, 168, "es_poster2", 6))
light((300, -240, 140), (255, 230, 190), 140); light((600, -240, 140), (255, 230, 190), 140, style=6)
# --- berber, çay ocağı, S1, emanet, kuaför, büfe, S2, geçit dükkânları
W(box(-600, -84, 0, -480, -80, 160, "es_wall", align="local"))                         # berber kapı açıklığı x -480..-400
W(plates(VOIDS["BERBER"], "es_wtile", ("-x", "-y")))
group("berber", shadow=True)
D(box(-596, -240, 0, -540, -120, 36, "es_counter", align="local"), relief("+x", -596, -240, -120, 44, 108, "es_mirror", 4),
  box(-500, -200, 0, -460, -160, 48, "es_seat", align="local"))
group("tabela", shadow=True)
D(relief("-y", -80, -600, -480, 118, 150, "es_s_berber", 6))
light((-500, -180, 140), (255, 240, 220), 110)
# çay ocağı: tezgâh ve semaver
group("cay", shadow=True)
D(box(-300, -120, 0, -160, -96, 44, "es_counter", align="local"), relief("+y", -96, -290, -170, 44, 100, "es_tea", 8))
D(box(-296, -160, 44, -164, -156, 160, "es_wall", align="local") if False else None)
group("tabela", shadow=True)
D(relief("-y", -80, -300, -172, 118, 150, "es_s_cay", 6))
light((-230, -120, 140), (255, 220, 170), 120)
sound((-230, -120, 80), "esenler/kalabalik.wav", 3, 2)
# S1: açık, boş dükkân (kepenk yarı açık: üstte plaka)
W(box(-700, -356, 100, -520, -352, 176, "es_shutter", align="local"))
group("dukkan", shadow=True)
D(box(-696, -200, 0, -600, -164, 60, "es_boxes", align="local"), relief("-x", -520, -300, -200, 40, 168, "es_poster1", 6))
light((-610, -250, 140), (255, 240, 220), 70, style=10)
# emanet
W(box(-760, 80, 0, -700, 84, 160, "es_wall", align="local"), box(-640, 80, 0, -560, 84, 160, "es_wall", align="local"))
group("emanet", shadow=True)
D(box(-756, 150, 0, -564, 180, 44, "es_counter", align="local"), box(-756, 180, 0, -564, 236, 140, "es_shelf", align="local", fit="uv"))
group("tabela", shadow=True)
D(relief("+y", 80, -760, -632, 118, 150, "es_s_emanet", 6))
light((-660, 120, 140), (255, 240, 220), 100)
# kuaför
W(box(200, 348, 0, 260, 352, 160, "es_wall", align="local"), box(340, 348, 0, 400, 352, 160, "es_wall", align="local"))
group("kuafor", shadow=True)
D(box(204, 200, 0, 396, 230, 36, "es_counter", align="local"), relief("+y", 196, 220, 380, 44, 108, "es_mirror", 4), box(280, 260, 0, 320, 300, 48, "es_seat", align="local"))
group("tabela", shadow=True)
D(relief("-y", 352, 200, 328, 118, 150, "es_s_kuafor", 6))
light((300, 270, 140), (255, 230, 240), 110)
# büfe
W(box(500, 80, 0, 540, 84, 160, "es_wall", align="local"), box(620, 80, 0, 700, 84, 160, "es_wall", align="local"))
group("bufe", shadow=True)
D(box(504, 180, 0, 696, 236, 140, "es_kiosk", align="local", fit="uv"), box(504, 140, 0, 696, 180, 44, "es_counter", align="local"))
group("tabela", shadow=True)
D(relief("+y", 80, 540, 668, 118, 150, "es_s_bufe", 6), relief("+y", 84, 710, 774, 30, 150, "es_sc", 6))        # sicakcikolata.com afişi büfe yanında
light((600, 130, 140), (255, 240, 220), 120)
# S2: C3 kuzeyinde depo odası (kapalı kepenk yarı)
W(box(-700, 512, 100, -500, 516, 176, "es_shutter2", align="local"))
group("dukkan", shadow=True)
D(box(-690, 600, 0, -600, 660, 50, "es_crate", align="local"), box(-560, 620, 0, -510, 668, 40, "es_sack", align="local"))
light((-600, 600, 140), (255, 240, 220), 60, style=10)
# geçit dükkânları N1, N2: içinde raf/koli (siper)
group("dukkan", shadow=True)
D(box(-396, 180, 0, -340, 260, 100, "es_shelf", align="local", fit="uv"), box(-300, 200, 0, -250, 250, 40, "es_boxes", align="local"))
D(box(764, 150, 0, 800, 300, 100, "es_shelf", align="local", fit="uv"), box(830, 200, 0, 876, 250, 44, "es_crate", align="local"))
light((-320, 216, 140), (255, 240, 220), 90, style=10); light((820, 216, 140), (255, 240, 220), 90)
# --- CT merdiven holü: bozuk yürüyen merdivenler + basamak; sahanlık; kapalı kapı
W(floor_slab(VOIDS["CTH"], "es_terrazzo"), plates(VOIDS["CTH"], "es_wall_up", ("-x", "+x")), plates(VOIDS["CTL"], "es_wall_up", ("-x", "+x", "+y")))
W(floor_slab((-200, 900, 192, 200, 1100, 368), "es_terrazzo"))
W(ramp("y", 512, 900, -200, -72, 0, 192, {"top": "es_escal", "slope": "es_escal", "side": "es_metal"}),
  ramp("y", 512, 900, 72, 200, 0, 192, {"top": "es_escal", "slope": "es_escal", "side": "es_metal"}))
W(stairs("y", 512, 900, -64, 64, 0, 192, {"top": "es_terrazzo", "side": "es_concrete"}, 16))
group("korkuluk")
for xk in (-72, -64, 64, 72):
    D(Brush([(xk - 3, 512, 0), (xk + 3, 512, 0), (xk - 3, 900, 192), (xk + 3, 900, 192), (xk - 3, 512, 40), (xk + 3, 512, 40), (xk - 3, 900, 232), (xk + 3, 900, 232)], "es_metal"))
group("kapi", shadow=True)
D(box(-120, 1092, 192, 120, 1096, 352, "es_door_gl", align="local", fit="uv"), box(-140, 1080, 192, 140, 1088, 232, "es_barrier", align="local"))
group("tabela", shadow=True)
D(relief("-y", 1092, -64, 64, 300, 332, "es_s_kapali", 6), relief("-y", 1092, -128, 128, 340, 372, "es_s_girilmez", 6) if False else None)
D(relief("-y", 1092, -128, 0, 280, 312, "es_s_cikis", 6) if False else None)
D(relief("-y", 1092, -64, 64, 250, 282, "es_s_cikis", 6))
D(relief("+x", -196, 600, 856, 100, 132, "es_s_metrook", 6) if False else None)
D(relief("-x", 196, 560, 816, 100, 132, "es_s_peron", 6), relief("+x", -196, 940, 1004, 60, 188, "es_poster2", 6))
for y in (600, 760):
    fluorescent(-32, y - 8, 32, y + 8, 368, brightness=120)
fluorescent(-32, 1000, 32, 1016, 368, brightness=130)
light((0, 1050, 300), (255, 244, 220), 60)
sound((0, 1000, 300), "esenler/kalabalik.wav", 4, 4)
sound((0, 1000, 300), "esenler/anons.wav", 7, 4 + 16 + 32, targetname="ses_anons")
sound((0, 1060, 300), "esenler/cigirtkan.wav", 6, 4 + 16 + 32, targetname="ses_cig")
# --- koridor kepenkleri ve tabelaları, afişler, moloz, bariyerler
group("kepenk", shadow=True)
SHUT = [("-y", -352, (-880, -752), "es_c2"), ("-y", -352, (-480, -352), None), ("-y", -352, (-300, -140), "es_s_bilet"),
        ("+y", -512, (-800, -660), None), ("+y", -512, (-520, -380), "es_c5"), ("+y", -512, (-250, -100), None), ("+y", -512, (200, 340), "es_c4"), ("+y", -512, (560, 700), None),
        ("+y", 80, (-1000, -900), None), ("-y", -80, (-900, -760), "es_c6"), ("+y", 80, (-520, -420), None), ("+y", 80, (200, 340), "es_c1"), ("+y", 80, (760, 870), None),
        ("-y", 352, (-880, -740), None), ("-y", 352, (-220, -100), "es_c3"), ("-y", 352, (460, 600), None), ("-y", 352, (640, 760), "es_c2"),
        ("+y", 512, (-440, -300), None), ("+y", 512, (260, 400), None), ("+y", 512, (600, 740), "es_c5")]
for face, c, (a0, a1), sgn in SHUT:
    D(relief(face, c, a0, a1, 0, 128, "es_shutter2" if (a0 // 100) % 3 == 0 else "es_shutter", 6))
    if sgn:
        D(relief(face, c, a0, a1, 132, 164 if sgn.startswith("es_s") else 164, sgn, 6))
for face, c, (a0, a1) in (("-x", -896, (-300, -236)), ("+x", 896, (100, 164)), ("+x", -1056, (-200, -136)), ("-x", 1056, (200, 264)), ("-x", 80, (-460, -396)), ("+x", -80, (400, 464))):
    D(relief(face, c, a0, a1, 40, 168, "es_poster1" if a0 % 128 else "es_poster2", 6))
D(relief("-x", 1056, -300, -236, 40, 168, "es_poster2", 6), relief("+x", -896, 200, 264, 40, 168, "es_poster1", 6))
group("moloz", shadow=True)
D(pyramid(ngon(1000, 470, 70, 8), 0, (1000, 470, 44), "es_rubble"), pyramid(ngon(-1000, -470, 60, 8), 0, (-1000, -470, 36), "es_rubble"))
D(box(880, 500, 0, 1056, 512, 176, "es_rubble", align="local") if False else None)
D(box(900, 440, 0, 1056, 512, 60, "es_rubble", align="local"))
D(box(-1052, 420, 0, -1020, 512, 44, "es_barrier", align="local") if False else None)
group("bariyer", shadow=True)
D(box(1056, -176, 20, 1060, -96, 44, "es_barrier", align="local") if False else None)
D(box(900, -512, 20, 1052, -508, 44, "es_barrier", align="local") if False else None)
D(box(-1056, -300, 20, -1052, -220, 44, "es_barrier", align="local") if False else None)
D(box(-896, -512, 20, -892, -352, 44, "es_barrier", align="local") if False else None)
D(box(-800, 352, 0, -796, 420, 44, "es_barrier", align="local"), box(820, -352, 0, 824, -280, 44, "es_barrier", align="local") if False else None)
# koridor eşyaları: bank, çöp kovası, koli yığınları (siper)
group("esya", shadow=True)
D(box(-300, 460, 0, -140, 484, 24, "es_seat", align="local"), box(-300, 484, 24, -140, 492, 56, "es_seat", align="local"))
D(box(300, -500, 0, 380, -440, 50, "es_boxes", align="local"), box(-150, -440, 0, -90, -380, 40, "es_crate", align="local"))
D(box(-980, 100, 0, -920, 160, 44, "es_crate", align="local"), box(960, -60, 0, 1030, 10, 56, "es_boxes", align="local"))
D(box(20, 300, 0, 70, 350, 40, "es_sack", align="local"), box(-70, -200, 0, -20, -150, 36, "es_boxes", align="local"))
# koridor floresanları (bazıları titrer), tavan boru hatları
for x in range(-960, 1200, 192):
    if x < 1200:
        fluorescent(x - 40, -440, x + 40, -424, 176, brightness=110, flicker=(x in (-576, 576)))
for x in range(-960, 1057, 192):
    fluorescent(x - 40, -8, x + 40, 8, 176, brightness=110, flicker=(x == 192))
    fluorescent(x - 40, 424, x + 40, 440, 176, brightness=100, flicker=(x == -384))
for y in range(-320, 400, 192):
    fluorescent(-984, y - 8, -920, y + 8, 176, brightness=100)
    fluorescent(-8, y - 8, 8 + 0, y + 8, 176, brightness=100) if False else None
    fluorescent(-40, y - 8, 40, y + 8, 176, brightness=100, flicker=(y == 64))
    fluorescent(936, y - 8, 1016, y + 8, 176, brightness=100)
group("boru")
D(box(-1056, -372, 160, 1200, -364, 168, "es_pipe", align="local"), box(-1056, 60, 160, 1056, 68, 168, "es_pipe", align="local"))
# sesler: havalandırma, floresan, uzak otobüs (avlu)
for pos in ((-500, -430, 150), (500, 0, 150), (-300, 430, 150), (1000, 0, 150), (-976, 0, 150)):
    sound(pos, "esenler/havalandirma.wav", 4, 4)
sound((-576, -430, 150), "esenler/floresan.wav", 5, 2); sound((192, 0, 150), "esenler/floresan.wav", 5, 2); sound((1300, -300, 120), "esenler/floresan.wav", 4, 2)
sound((-1300, -1100, 300), "esenler/otobus.wav", 6, 4); sound((-1200, -600, 200), "esenler/ruzgar.wav", 3, 4)
sound((-1300, -1100, 300), "esenler/fren.wav", 7, 4 + 16 + 32, targetname="ses_fren")
ents.append(({"classname": "multi_manager", "targetname": "mm_sesler", "spawnflags": "1",
              "ses_anons": "8", "ses_cig": "20", "ses_fren": "33", "mm_sesler": "48"}, []))
ent("trigger_auto", (0, 0, 64), target="mm_sesler", spawnflags=1)

# ================================================================ FİGÜRLER
figure(-230, -130, 2, 270, 1)                         # ikramcı (çay ocağı)
figure(90, 560, 2, 180, 3, blink=(12, 8))             # kayıp yolcu (merdiven dibi)
figure(690, -240, 2, 180, 0)                           # lokanta aşçısı (tezgâh arkası)
figure(-1300, 250, -62, 90, 2, blink=(10, 10))        # depo işçisi
figure(1120, -400, 2, 180, 1, blink=(14, 9))          # tuvalet kapısında bekleyen
figure(-1090, -1200, 194, 180, 3, blink=(11, 11))     # muavin (avluda kolilerin yanında)
figure(-960, 420, 2, 0, 0)                             # yazıhaneci (koridor köşesinde)

# ================================================================ OYUN VARLIKLARI
for i in range(16):
    ent("info_player_deathmatch", (-1520 + (i % 4) * 100, -1300 + (i // 4) * 80, 232), angles="0 0 0")
for i in range(16):
    ent("info_player_start", (-150 + (i % 4) * 100, 940 + (i // 4) * 40, 232), angles="0 270 0")
ent("func_bomb_target", brushes=[box(1200, -512, -40, 1600, -80, 150, "aaatrigger")])        # A: tuvalet
ent("func_bomb_target", brushes=[box(-1600, 80, -70, -1056, 512, 150, "aaatrigger")])        # B: depo
ent("info_map_parameters", (0, 0, 600), bombradius=300)
ent("light_environment", (-1300, -1100, 600), angles="0 250 0", pitch=-60, _light="255 248 236 220", _diffuse_light="190 205 230 170")

if __name__ == "__main__":
    write(os.path.join(HERE, "de_esenler.map"), "esenler", "Esenler Otogari Alt Kat", 6144)
