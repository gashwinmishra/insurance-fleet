from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder(text, filename, color):
    img = Image.new('RGB', (600, 400), color=color)
    d = ImageDraw.Draw(img)
    # Drawing simple text - positioned roughly
    d.text((50, 180), text, fill=(255, 255, 255))
    img.save(f"insurance-fleet/assets/{filename}")
    print(f"Created {filename}")

if __name__ == "__main__":
    os.makedirs("insurance-fleet/assets", exist_ok=True)
    
    # Failed AI images - creating placeholders
    scenarios = [
        ("Undercarriage Scrape\n(Road Debris)", "ev_undercarriage_scrape_road_debris.png", (80, 80, 80)),
        ("Fender Paint Transfer\n(Scuff)", "ev_fender_paint_transfer.png", (100, 100, 200)),
        ("HV Cable Corrosion\n(Orange Connector)", "ev_battery_corrosion_connector.png", (200, 100, 50)),
        ("Wheel Curb Rash\n(Alloy Damage)", "ev_wheel_curb_rash.png", (50, 50, 50)),
        ("Windshield Chip\n(Star crack)", "ev_windshield_chip.png", (100, 200, 250)),
        ("Frunk Dent\n(Hail)", "ev_frunk_dented_hood.png", (70, 70, 90)),
        ("Missing Shingles\n(Wind Damage)", "roof_missing_shingles_wind.png", (120, 100, 80)),
        ("Rusted Flashing\n(Chimney)", "roof_flashing_rusted_chimney.png", (150, 80, 80))
    ]
    
    for text, filename, color in scenarios:
        create_placeholder(text, filename, color)
