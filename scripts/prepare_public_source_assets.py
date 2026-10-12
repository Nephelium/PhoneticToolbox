"""Supply the application icon when a public checkout lacks the private avatar."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / 'frontend/src/assets/feedback-avatar.jpg'
    if target.exists():
        print('Existing author avatar retained.')
        return
    from PIL import Image
    source = ROOT / 'frontend/src/assets/k2.png'
    with Image.open(source) as image, target.open('xb') as output:
        image.convert('RGB').save(output, 'JPEG', quality=95)
    print('Public source checkout uses the application icon as its feedback avatar.')


if __name__ == '__main__':
    main()
