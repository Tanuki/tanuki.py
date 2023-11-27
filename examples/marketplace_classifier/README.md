# Marketplace Product Auto-Tagger using Tanuki

Example of how Tanuki can be used to auto-tag products based on their description and name for a marketplace app.

## Installation

1. Install dependencies from requirements.txt

```bash
pip install -r requirements.txt
```

2. Create a `.env` file based on the `.env.example` file and fill in the required information.

**Note**: You'll need an [OpenAI API key](https://openai.com/blog/openai-api) to use this. *Also note*, this demo uses the [BestBuy Product API](https://bestbuyapis.github.io/api-documentation/#products-api), but you can use any API you want. Just make sure to update the code accordingly. If you want to use BestBuy, you'll need to create an account and get an API key from them.

3. Run the app

```bash
python main.py
```

## Examples

Original input:

```
Product: Comma [LP] - VINYL
Short Description: None
Description: None
Long Description: None
```

Generated output:
```python
{
    'category': 'Music & Audio', 
    'brand': 'unknown', 
    'features': 'VINYL'
}
```

<hr />

Original input:

```
    Product: 4-Year Standard Geek Squad Protection
    
    Short Description: Enhance your manufacturer warranty and get extended coverage when the warranty ends, including:
    If your original bulb burns out, we'll replace it. If there's a failure from normal wear and tear, we'll repair it. 
    If your projector won't work because of a power surge, we'll fix it. You'll never pay for parts and labor on covered 
    repairs.
    
    Description: None
    
    Long Description: You're excited to buy a new projector for your home &#8212; now, make sure to protect it. Geek 
    Squad&#174; Protection enhances your manufacturer warranty and gives you extended coverage when the warranty ends.  
    Geek Squad Protection often covers important repairs that your manufacturer warranty doesn't (like failure due to a 
    power surge).  If something goes wrong after the manufacturer warranty ends, rest easy; you won't be stuck with a 
    huge repair bill. Geek Squad Protection has your back and will take care of any covered repairs.
```

Generated output:
```python
{
    'category': 'Electronics', 
    'brand': 'Geek Squad', 
    'features': 'Extended warranty, Coverage for bulb burnout, Coverage for normal wear and tear, Coverage for power surge'
}
```

<hr />

```
    Product: Linon Home Décor - Kessler Two-Tone Childrens Bookcase - White and Natural
    
    Short Description: None
    
    Description: None
    
    Long Description: Keeping your little one&#8217;s play space organized is a breeze with this bookcase. A fun and 
    functional farmhouse look will encourage your child to keep their space tidy. An ample top surface, an upper shelf, 
    and a lower bin accented with a fence design are perfectly sized to fit all your child&#8217;s books, stuffed 
    animals, and toys. The combination of a white finish with natural pine wood accents will combine well with any 
    d&#233;cor. Solidly constructed square peg legs provide support on any flat surface.
```
    

Generated output:
```python
{
    'category': 'Home Goods', 
    'brand': 'Linon Home Décor', 
    'features': 'Two-Tone, White and Natural, upper shelf, lower bin, white finish with natural pine wood accents, Solidly constructed square peg legs'
}
```


<hr />

```
    Product: Razer - Kraken Kitty Edition V2 Pro Wired Gaming Headset - Black
    
    Short Description: None
    
    Description: None
    
    Long Description: Creating the cutest stream persona now comes in more ways than one. Switch up your style and light 
    up your stream with a Razer Chroma RGB headset featuring 3 interchangeable ear designs, stream reactive lighting for 
    next-level audience engagement, and a solid mic for crystal-clear voice capture.
```

Generated output:
```python
{
    'category': 'Electronics', 
    'brand': 'Razer', 
    'features': 'Razer Chroma RGB, interchangeable ear designs, stream reactive lighting, solid mic for crystal-clear voice capture'
}
```