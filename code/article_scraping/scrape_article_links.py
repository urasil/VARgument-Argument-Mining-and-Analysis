from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException, TimeoutException
import os
import time

class SkySportsScraper:
    def __init__(self, url, output_file):
        self.url = url
        self.output_file = output_file
        self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(100)
        self.driver.get(self.url)
        self.click_cookie_consent()
       
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w') as file:
                file.write("title,href\n")

    def click_cookie_consent(self):
        try:
            WebDriverWait(self.driver, 10).until(
                EC.frame_to_be_available_and_switch_to_it((By.ID, "sp_message_iframe_1168576"))
            )
            print("Switched to iframe")
            cookie_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Essential cookies only']"))
            )
            cookie_button.click()
            print("Cookie consent button clicked")
            self.driver.switch_to.default_content()
            print("Switched back to main content")

        except TimeoutException:
            print("Cookie consent button not found or timeout occurred")
        except ElementClickInterceptedException:
            print("Cookie consent button click intercepted, using JavaScript click")
            self.driver.execute_script("arguments[0].click();", cookie_button)

    def write_to_file(self, aria_label, href):
        with open(self.output_file, 'r') as file:
            existing_articles = file.readlines()

        entry = f"{aria_label},{href}\n"

        if entry not in existing_articles:
            with open(self.output_file, 'a') as file:
                file.write(entry)

    def scrape_articles(self):
        while True:
            articles = self.driver.find_elements(By.CSS_SELECTOR, "div.news-list__item.news-list__item--show-thumb-bp30")

            if not articles:
                print("No articles found")
                break

            for article in articles:
                title = article.find_element(By.TAG_NAME, "a").get_attribute("title")
                new_title = title.replace(",", ";")
                # print("title: ", title)
                href = article.find_element(By.TAG_NAME, "a").get_attribute("href")

                self.write_to_file(new_title, href)

            try:
                show_more_button = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a[data-role='load-more']"))
                )
                self.driver.execute_script("arguments[0].scrollIntoView(true);", show_more_button)

                WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a[data-role='load-more']"))
                )

                show_more_button.click()

                time.sleep(2)  # wait for more articles to load
            except Exception as e:
                print("No more 'Show More' button found or an error occurred:", e)
                break


    def close_driver(self):
        self.driver.quit()

# Usage
if __name__ == "__main__":
    url = "https://www.skysports.com/premier-league-news"
    output_file = "football_articles.txt"

    scraper = SkySportsScraper(url, output_file)
    scraper.scrape_articles()
    scraper.close_driver()
