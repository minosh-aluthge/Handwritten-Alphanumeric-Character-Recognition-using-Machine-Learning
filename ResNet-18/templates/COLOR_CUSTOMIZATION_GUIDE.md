# 🎨 Color Customization Guide

## Easy Color Palette Editing

All colors are now defined at the **top of the CSS** using CSS variables. You only need to edit **ONE SECTION** to change the entire color scheme!

## 📍 Location

Open: `templates/index.html`  
Find: Lines 9-22 (right after `<style>`)

## 🎨 Your Current Color Palette

```css
:root {
    --color-primary: #ECFEAA;     /* Light lime green */
    --color-secondary: #303030;   /* Dark gray */
    --color-accent-1: #4D4D4D;    /* Medium gray */
    --color-accent-2: #808080;    /* Light gray */
    --color-accent-3: #F79F79;    /* Coral orange */
    --color-accent-4: #E5625E;    /* Red coral */
}
```

## 🖌️ Color Usage Map

| Variable | Used For | Examples |
|----------|----------|----------|
| `--color-primary` | Primary accents | Upload border, highlights |
| `--color-secondary` | Main text | Headings, dark text |
| `--color-accent-1` | Borders | Canvas border |
| `--color-accent-2` | (Reserved) | Future use |
| `--color-accent-3` | Interactive elements | Buttons, links, predictions |
| `--color-accent-4` | Secondary interactive | Button borders, confidence |

## 🎯 How to Change Colors

### Method 1: Edit the Palette (Recommended)

Just change the hex codes in the `:root` section:

```css
:root {
    --color-primary: #YOUR_COLOR_1;
    --color-secondary: #YOUR_COLOR_2;
    --color-accent-1: #YOUR_COLOR_3;
    --color-accent-2: #YOUR_COLOR_4;
    --color-accent-3: #YOUR_COLOR_5;
    --color-accent-4: #YOUR_COLOR_6;
}
```

**That's it!** The entire site will update automatically.

### Method 2: Use a Different Palette

Here are some pre-made palettes you can copy & paste:

#### 🌊 Ocean Blue
```css
:root {
    --color-primary: #A8DADC;     /* Powder blue */
    --color-secondary: #1D3557;   /* Navy */
    --color-accent-1: #457B9D;    /* Steel blue */
    --color-accent-2: #6C9AB5;    /* Sky blue */
    --color-accent-3: #F1FAEE;    /* Off white */
    --color-accent-4: #E63946;    /* Red */
}
```

#### 🌸 Pastel Pink
```css
:root {
    --color-primary: #FFD6E8;     /* Light pink */
    --color-secondary: #4A4A4A;   /* Dark gray */
    --color-accent-1: #FFADD2;    /* Pink */
    --color-accent-2: #FF85B3;    /* Hot pink */
    --color-accent-3: #C77DFF;    /* Purple */
    --color-accent-4: #9D4EDD;    /* Dark purple */
}
```

#### 🌿 Forest Green
```css
:root {
    --color-primary: #B7E4C7;     /* Mint green */
    --color-secondary: #1B4332;   /* Dark green */
    --color-accent-1: #52B788;    /* Green */
    --color-accent-2: #74C69D;    /* Light green */
    --color-accent-3: #40916C;    /* Forest green */
    --color-accent-4: #2D6A4F;    /* Deep green */
}
```

#### 🔥 Sunset Orange (Current)
```css
:root {
    --color-primary: #ECFEAA;     /* Light lime green */
    --color-secondary: #303030;   /* Dark gray */
    --color-accent-1: #4D4D4D;    /* Medium gray */
    --color-accent-2: #808080;    /* Light gray */
    --color-accent-3: #F79F79;    /* Coral orange */
    --color-accent-4: #E5625E;    /* Red coral */
}
```

#### 🌌 Dark Mode
```css
:root {
    --color-primary: #BB86FC;     /* Purple */
    --color-secondary: #FFFFFF;   /* White */
    --color-accent-1: #3700B3;    /* Deep purple */
    --color-accent-2: #6200EE;    /* Purple */
    --color-accent-3: #03DAC6;    /* Teal */
    --color-accent-4: #018786;    /* Dark teal */
}
```

## 🎨 Finding Your Own Colors

### Online Tools:
1. **Coolors.co** - Generate color palettes
2. **Adobe Color** - Create harmonious colors
3. **Material Design Colors** - Google's color system

### Tips:
- Use 2-3 main colors + 2-3 accent colors
- Ensure good contrast for readability
- Test on both light and dark backgrounds

## 🔄 Quick Testing

After editing:
1. Save `index.html`
2. Refresh your browser (Ctrl+F5 / Cmd+Shift+R)
3. See changes instantly!

## 📊 Visual Reference

Your current palette visualization:

```
ECFEAA  303030  4D4D4D  808080  F79F79  E5625E
████    ████    ████    ████    ████    ████
Primary Secondary Accent-1 Accent-2 Accent-3 Accent-4
Lime    Dark    Medium   Light    Coral   Red
Green   Gray    Gray     Gray    Orange  Coral
```

## 🚀 Advanced: Gradient

The gradient is auto-generated from accent colors:
```css
--color-gradient: linear-gradient(135deg, var(--color-accent-3) 0%, var(--color-accent-4) 100%);
```

Change `135deg` to adjust gradient angle:
- `0deg` - Bottom to top
- `90deg` - Left to right
- `135deg` - Diagonal (current)
- `180deg` - Top to bottom

## ✅ Validation Checklist

After changing colors:
- [ ] Text is readable on all backgrounds
- [ ] Buttons stand out clearly
- [ ] Hover effects are visible
- [ ] Prediction results are easy to read
- [ ] Canvas border is visible

## 💡 Pro Tips

1. **Consistency**: Use the same colors throughout
2. **Contrast**: Light text on dark, dark text on light
3. **Accessibility**: Check color contrast ratios
4. **Testing**: Test on different devices/browsers

---

**Last Updated:** October 19, 2025  
**Current Theme:** Sunset Orange (Coral + Lime)
