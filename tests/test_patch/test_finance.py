from typing import List

from dotenv import load_dotenv

load_dotenv()
import tanuki


@tanuki.patch
def extract_stock_winners_vol6(input: str) -> List[str]:
    """
    Below you will find an article with stocks analysis. Bring out the stock symbols of companies who are expected to go up or have positive sentiment
    """

@tanuki.align
def align_classify_sentiment():
    """We can test the function as normal using Pytest or Unittest"""

    input_1 = "Consumer spending makes up a huge fraction of the overall economy. Investors are therefore always looking at consumers to try to gauge whether their financial condition remains healthy. That's a big part of why the stock market saw a bear market in 2022, as some feared that a consumer-led recession would result in much weaker business performance across the sector.\nHowever, that much-anticipated recession hasn't happened yet, and there's still plenty of uncertainty about the future direction of consumer-facing stocks. A pair of earnings reports early Wednesday didn't do much to resolve the debate, as household products giant Procter & Gamble (PG 0.13%) saw its stock rise even as recreational vehicle manufacturer Winnebago Industries (WGO 0.58%) declined."
    assert extract_stock_winners_vol6(input = input_1) ==["Procter & Gamble", "Winnebago Industries"] 


def test_classify_sentiment():
    align_classify_sentiment()
    input = "A recent survey by Nationwide, the financial services firm, found that over three-quarters of both Gen Z and millennials expect they will need to continue working into their retirement years because they do not believe Social Security will be enough to rely on in their old age.\nIt's a troubling situation, but the good news is that if you invest in dividend stocks, they can help strengthen your prospects for retirement. Not only can these types of investments increase the value of your portfolio over time, but they will also provide you with recurring cash flow.\nThree dividend stocks that can be excellent investments to include as part of your retirement plan now are UnitedHealth Group (UNH -0.26%), Verizon Communications (VZ 0.83%), and ExxonMobil (XOM 1.31%)."
    output = extract_stock_winners_vol6(input)
    assert "Verizon Communications" in output and "ExxonMobil" in output and "UnitedHealth Group" in output