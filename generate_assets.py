from PIL import Image, ImageDraw, ImageFont
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")

def create_placeholder(text, filename, color):
    img = Image.new('RGB', (600, 400), color=color)
    d = ImageDraw.Draw(img)
    # Drawing simple text
    d.text((50, 180), text, fill=(255, 255, 255))
    os.makedirs(assets_dir, exist_ok=True)
    img.save(os.path.join(assets_dir, filename))
    print(f"Created {filename}")

if __name__ == "__main__":
    create_placeholder("Premium EV\n(Scratch on Battery)", "car_scratch.jpg", (50, 50, 100))
    create_placeholder("Roof Inspection\n(Granule Loss)", "roof_damage.jpg", (100, 50, 50))
