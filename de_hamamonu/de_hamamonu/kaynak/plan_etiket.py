"""prev_plan.png üstüne bölge etiketleri yazar -> onizleme/plan.png"""
import os, sys, math
from PIL import Image, ImageDraw, ImageFont

here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "onizleme", "prev_plan.png")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "onizleme", "plan.png")
im = Image.open(src).convert("RGB")
W, H = im.size
CX, CY = -128, 0                      # render: eye (-128,-1,12000) -> (-128,0,0), fov 21
half = math.tan(math.radians(10.5)) * 12000
sx = W / 2 / half
P = lambda x, y: (W / 2 + (x - CX) * sx, H / 2 - (y - CY) * sx)
d = ImageDraw.Draw(im, "RGBA")
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44)
fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
def label(x, y, text, font=f, fill=(255, 255, 255)):
    bb = d.textbbox((0, 0), text, font=font)
    px, py = P(x, y)
    px -= (bb[2] - bb[0]) / 2; py -= (bb[3] - bb[1]) / 2
    for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
        d.text((px + ox, py + oy), text, font=font, fill=(0, 0, 0))
    d.text((px, py), text, font=font, fill=fill)
def zone(x0, y0, x1, y1, rgb):
    d.rectangle([P(x0, y1), P(x1, y0)], fill=rgb + (70,), outline=rgb + (255,), width=3)
zone(-1200, -200, -840, 200, (240, 200, 40)); label(-1024, -260, "B", f2, (255, 220, 60))
zone(760, -300, 1200, 140, (230, 60, 40)); label(960, -220, "A", f2, (255, 90, 70))
zone(-1330, -1110, -1060, -870, (230, 170, 40)); label(-1195, -990, "T", f2, (255, 200, 60))
zone(750, 1090, 1050, 1330, (60, 130, 240)); label(900, 1210, "CT", f2, (120, 170, 255))
label(-1024, 330, "Saat Kulesi Meydanı"); label(-1024, -330, "çay bahçesi", fs)
label(960, 420, "Hacettepe Önü"); label(960, 830, "HACETTEPE (lobi geçilir)", fs)
label(-64, 352, "ANA SOKAK (üst kot)"); label(-300, 896, "kuzey sokak (üst kot)", fs)
label(-320, -400, "Hamam"); label(-128, -128, "avlu", fs); label(112, -400, "Bakırcı (ev)", fs)
label(-128, 100, "geçilen evler", fs); label(304, -300, "merdivenli sokak", fs); label(-576, -200, "Hamam Sokağı", fs)
label(-1040, 520, "Kule Yokuşu", fs); label(-768, 290, "büyük merdiven", fs)
label(-224, 640, "cumbalı evler", fs); label(0, -688, "güney sokak (alt kot)", fs)
label(960, -480, "merdiven", fs); label(576, 1070, "yan geçit", fs); label(1456, 1070, "yan geçit", fs)
im.save(dst)
print("yazıldı", dst)
