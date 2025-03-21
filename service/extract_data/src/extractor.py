from lxml import html 
from typing import Dict, List

class Extractor:
    def __init__(self, response: str):
        self.response = response
        self.tree = html.fromstring(response.text)
        
    def extract_element(self) -> List[Dict]:
        
        category = self.response.url.split("/")[3]
        title = self.tree.xpath("//div[@class='box_type_m']//table[1]//tr[position() >= 3]//td//a/text()")
        links = self.tree.xpath("//div[@class='box_type_m']//table[1]//tr[position() >= 3]//td[1]//a/@href") 
        pdf_files = self.tree.xpath("//div[@class='box_type_m']//table[1]//tr[position() >= 3]//td[@class='file']//a/@href") 
        stocks = self.tree.xpath("//div[@class='box_type_m']//table[1]//tr[position() >= 3]//td[2]/text()") 
        date = self.tree.xpath("//div[@class='box_type_m']//table[1]//tr[position() >= 3]//td[@class='date'][1]/text()") 
        
        result = []
        for title, link, pdf, stock, date, cate in zip(title, links, pdf_files, stocks, date, category):
            full_links = f"https://finance.naver.com/{category.strip()}/{link.strip()}"
            result.append({
                "title": title.strip(),
                "link": str(full_links),
                "pdf": pdf if pdf else None,
                "stock": stock.strip(),
                "date": f"20" + date.strip()
                })
        
        return result 
    
    
    def extract_industry_stockitems(self) -> List[Dict]:
        
        category = self.response.url.split("/")[3]
        item = self.tree.xpath("//div[@class='box_type_m']//table[1]//tr[position() >= 3]//td[@style='padding-left:10']//a/text()")
        title = self.tree.xpath("//div[@class='box_type_m']//table[1]//tr[position() >= 3]//td[2]//a/text()") 
        links = self.tree.xpath("//div[@class='box_type_m']//table[1]//tr[position() >= 3]//td[2]//a/@href") 
        pdf_files = self.tree.xpath("//div[@class='box_type_m']//table[1]//tr[position() >= 3]//td[@class='file']//a/@href") 
        stocks = self.tree.xpath("//div[@class='box_type_m']//table[1]//tr[position() >= 3]//td[3]/text()") 
        date = self.tree.xpath("//div[@class='box_type_m']//table[1]//tr[position() >= 3]//td[@class='date'][1]/text()") 
        
        result = []
        
        for item, title, link, pdf, stock, date in zip(item, title, links, pdf_files, stocks, date):
            full_links = f"https://finance.naver.com/{category.strip()}/{link.strip()}"
            result.append({
                "items" : item.strip(),
                "title": title.strip(),
                "link": str(full_links),
                "pdf": pdf if pdf else None,
                "stock": stock.strip(),
                "date": f"20"+date.strip()
                })
            
        return result 
    
    def detail_page_crawler(self) -> str:  #임시 타입
        main_content = self.tree.xpath("//td[@class='view_cnt']//p//text()")
        
        return main_content        