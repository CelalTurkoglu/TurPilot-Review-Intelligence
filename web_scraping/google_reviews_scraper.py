import csv
import importlib
import random
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    SessionNotCreatedException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Buraya 8-10 adet Google Isletme Yorumlari linkini ekleyin.
# Ornek: "https://www.google.com/maps/place/.../data=..."
GOOGLE_REVIEW_LINKS = [
        "https://www.google.com/search?q=gezilife&num=10&sca_esv=8c24cb2666462953&hl=tr-TR&biw=1470&bih=801&tbm=lcl&sxsrf=ANbL-n6lOxa9IaCdsWn0XVNJUywAuH1oTA%3A1777931625190&ei=aRX5acGtC72P7NYP69_ugAc#lkt=LocalPoiReviews&rlfi=hd:;si:7259563926817306114,l,CghnZXppbGlmZUjm2O-Bv7iAgAhaFBAAGAAiCGdlemlsaWZlKgQIAhAAkgENdHJhdmVsX2FnZW5jeQ;mv:[[41.290222899999996,36.3377876],[41.2847836,36.326096799999995]]",
        "https://www.google.com/search?q=gezione&num=10&sca_esv=8c24cb2666462953&hl=tr-TR&biw=1470&bih=801&tbm=lcl&sxsrf=ANbL-n75sCckL25m3gAXHSPRvxcN3pjEQg%3A1777931629782&ei=bRX5af2yL5eUxc8Pt8uz2Aw#lkt=LocalPoiReviews&rlfi=hd:;si:11528282663559099428,l,CgdnZXppb25lSIfzzf34uoCACFoTEAAYACIHZ2V6aW9uZSoECAIQAJIBF3NpZ2h0c2VlaW5nX3RvdXJfYWdlbmN5,y,Jbd8iG87Nj4;mv:[[41.2901965,36.3413879],[41.285249199999996,36.325893]]",
        "https://www.google.com/search?q=city+turizm&num=10&sca_esv=8c24cb2666462953&hl=tr-TR&biw=1470&bih=801&tbm=lcl&sxsrf=ANbL-n4dfIMpHi-U9XquDh21wU-iPvdl-A%3A1777931661195&ei=jRX5aaXPC9Hmxc8P8ZuImQc#lkt=LocalPoiReviews&rlfi=hd:;si:18433514296283617995,l,CgtjaXR5IHR1cml6bUibpoGi54CAgAhaHRAAEAEYABgBIgtjaXR5IHR1cml6bSoGCAIQABABkgENdHJhdmVsX2FnZW5jeQ;mv:[[40.98796777731903,28.884052028115086],[40.98760782268097,28.88357517188491]]",
        "https://www.google.com/search?q=turizm+acentesi&num=10&sca_esv=8c24cb2666462953&hl=tr-TR&biw=1470&bih=801&tbm=lcl&sxsrf=ANbL-n6VViFy5mJ6Btkn2uloDakEuHjkJQ%3A1777931675762&ei=mxX5adWbLraG7NYPxMWk2Qc#lkt=LocalPoiReviews&rlfi=hd:;si:2974552164502804418,l,Cg90dXJpem0gYWNlbnRlc2lIjqyDjPKqgIAIWhUQABABIg90dXJpem0gYWNlbnRlc2mSAQ10cmF2ZWxfYWdlbmN5mgEkQ2hkRFNVaE5NRzluUzBWSlEwRm5TVVJpYjNOUWFqWjNSUkFC-gEECAAQPQ;mv:[[41.344320200000006,36.3596905],[41.270998999999996,36.2559958]]",
        "https://www.google.com/search?q=turizm+acentesi&num=10&sca_esv=8c24cb2666462953&hl=tr-TR&biw=1470&bih=801&tbm=lcl&sxsrf=ANbL-n6VViFy5mJ6Btkn2uloDakEuHjkJQ%3A1777931675762&ei=mxX5adWbLraG7NYPxMWk2Qc#lkt=LocalPoiReviews&rlfi=hd:;si:1304472050969966318,l,Cg90dXJpem0gYWNlbnRlc2lIu5jMl4ergIAIWhcQABABGAAiD3R1cml6bSBhY2VudGVzaZIBDXRyYXZlbF9hZ2VuY3maAURDaTlEUVVsUlFVTnZaRU5vZEhsalJqbHZUMjFGZUZoNlNtWmlSMFp5VFVoc1dtRllTVFJaYlU1YVZVVnNhbUl5WXhBQvoBBAgAED4;mv:[[41.344320200000006,36.3596905],[41.270998999999996,36.2559958]]",
        "https://www.google.com/search?q=turizm+acentesi&num=10&sca_esv=8c24cb2666462953&hl=tr-TR&biw=1470&bih=801&tbm=lcl&sxsrf=ANbL-n6VViFy5mJ6Btkn2uloDakEuHjkJQ%3A1777931675762&ei=mxX5adWbLraG7NYPxMWk2Qc#lkt=LocalPoiReviews&rlfi=hd:;si:2633763629111957420,l,Cg90dXJpem0gYWNlbnRlc2lI-Pik4v-4gIAIWhUQABABIg90dXJpem0gYWNlbnRlc2mSARdzaWdodHNlZWluZ190b3VyX2FnZW5jeZoBRENpOURRVWxSUVVOdlpFTm9kSGxqUmpsdlQycFdUbFl4WkU5TVZGb3pZM2t3ZEZGdVVraE9SRnA1VWpGWk5FMVdSUkFC-gEECAAQSQ;mv:[[41.344320200000006,36.3596905],[41.270998999999996,36.2559958]]",
        "https://www.google.com/search?q=karab%C3%BCk+turizm+acentesi&num=10&sca_esv=8c24cb2666462953&hl=tr-TR&biw=1470&bih=801&tbm=lcl&sxsrf=ANbL-n6ul6aPRJDatTV62ZFdW_tLcU6PzA%3A1777931694645&ei=rhX5aciQJ8qExc8PrLCskQ4#lkt=LocalPoiReviews&rlfi=hd:;si:4772385785205234766,l,ChhrYXJhYsO8ayB0dXJpem0gYWNlbnRlc2lI1fzSmKy6gIAIWiQQARACGAAYARgCIhhrYXJhYsO8ayB0dXJpem0gYWNlbnRlc2mSARdzaWdodHNlZWluZ190b3VyX2FnZW5jeZoBRENpOURRVWxSUVVOdlpFTm9kSGxqUmpsdlQycHNhMDFZV2xsVVIxcE5ZMWhzTlZkVmNFVlRNalZNVjJzMU1VMVlZeEFC-gEECAAQQg;mv:[[41.2543381,32.6976848],[41.190799,32.6056449]]",
        "https://www.google.com/search?q=ankara+turizm+acentesi&num=10&sca_esv=8c24cb2666462953&hl=tr-TR&biw=1470&bih=801&tbm=lcl&sxsrf=ANbL-n4dizjIxuyHPysaD7sJs5Pbc0IOHA%3A1777931787974&ei=Cxb5aeKUO_iP7NYP9fjcuAM#lkt=LocalPoiReviews&rlfi=hd:;si:9547286929434860130,a;mv:[[40.006791299999996,32.908120700000005],[39.8812102,32.4949595]]",
        "https://www.google.com/search?q=ankara+turizm+acentesi&num=10&sca_esv=8c24cb2666462953&hl=tr-TR&biw=1470&bih=801&tbm=lcl&sxsrf=ANbL-n4dizjIxuyHPysaD7sJs5Pbc0IOHA%3A1777931787974&ei=Cxb5aeKUO_iP7NYP9fjcuAM#lkt=LocalPoiReviews&rlfi=hd:;si:11084384151771416631,l,ChZhbmthcmEgdHVyaXptIGFjZW50ZXNpSLGwioz0roCACFoeEAEQAhgAIhZhbmthcmEgdHVyaXptIGFjZW50ZXNpkgENdHJhdmVsX2FnZW5jeZoBJENoZERTVWhOTUc5blMwVkpRMEZuU1VNM2JHSlBiakZSUlJBQvoBBAhHEEU;mv:[[40.006791299999996,32.908120700000005],[39.8812102,32.4949595]]",
        "https://www.google.com/search?q=ankara+turizm+acentesi&num=10&sca_esv=8c24cb2666462953&hl=tr-TR&biw=1470&bih=801&tbm=lcl&sxsrf=ANbL-n4dizjIxuyHPysaD7sJs5Pbc0IOHA%3A1777931787974&ei=Cxb5aeKUO_iP7NYP9fjcuAM#lkt=LocalPoiReviews&rlfi=hd:;si:7128466032902645080,l,ChZhbmthcmEgdHVyaXptIGFjZW50ZXNpSLGw6KK1t4CACFoiEAEQAhgAGAEYAiIWYW5rYXJhIHR1cml6bSBhY2VudGVzaZIBDXRyYXZlbF9hZ2VuY3maAURDaTlEUVVsUlFVTnZaRU5vZEhsalJqbHZUMmt4UmxKWFJsSlphMDVIVmxjeGExTkZkR3BhTWtwWFlXeFdXR1JzUlJBQvoBBAgqEEM;mv:[[40.006791299999996,32.908120700000005],[39.8812102,32.4949595]]",
        "https://www.google.com/search?q=bursa+turizm+acentesi&num=10&sca_esv=8c24cb2666462953&hl=tr-TR&biw=1470&bih=801&tbm=lcl&sxsrf=ANbL-n7dHueiTrgRkq3Ye3Z1yHYVw2gkKg%3A1777931882877&ei=ahb5abShNa2Vxc8Px7_LmQI#lkt=LocalPoiReviews&rlfi=hd:;si:4135155966505736124,l,ChVidXJzYSB0dXJpem0gYWNlbnRlc2lIpJyGi7OygIAIWh0QARACGAAiFWJ1cnNhIHR1cml6bSBhY2VudGVzaZIBDXRyYXZlbF9hZ2VuY3maAURDaTlEUVVsUlFVTnZaRU5vZEhsalJqbHZUMjE0YkdKc1RuQlZWMFpLVG0wMU0xWnFWbkZOUkd4WVlqRkdlbFF3UlJBQvoBBQjeAhBG;mv:[[40.2568571,29.075424199999997],[40.1781903,28.833067000000003]]",
        "https://www.google.com/search?q=bursa+turizm+acentesi&num=10&sca_esv=8c24cb2666462953&hl=tr-TR&biw=1470&bih=801&tbm=lcl&sxsrf=ANbL-n7dHueiTrgRkq3Ye3Z1yHYVw2gkKg%3A1777931882877&ei=ahb5abShNa2Vxc8Px7_LmQI#lkt=LocalPoiReviews&rlfi=hd:;si:8255094751721711918,l,ChVidXJzYSB0dXJpem0gYWNlbnRlc2lI0f2Ums-5gIAIWh8QARACGAAYASIVYnVyc2EgdHVyaXptIGFjZW50ZXNpkgENdHJhdmVsX2FnZW5jeZoBI0NoWkRTVWhOTUc5blMwVkpRMEZuU1VRM2VXTjJkMHRCRUFF-gEECAAQRw;mv:[[40.2568571,29.075424199999997],[40.1781903,28.833067000000003]]",
]

OUTPUT_FILE = Path(__file__).resolve().parent / "dataset.csv"
DEBUG_DIR = Path(__file__).resolve().parent / "debug"
USE_UNDETECTED_CHROME = False
TARGET_PER_STAR = 10
STAR_VALUES = [1, 2, 3, 4, 5]
LOW_STAR_VALUES = [1, 2, 3]
LOW_SORT_STAR_VALUES = [1, 2, 3, 4]
HIGH_STAR_VALUES = [5, 4]


def human_sleep(min_seconds=1.0, max_seconds=3.0):
    """Google tarafinda bot davranisini azaltmak icin rastgele bekler."""
    time.sleep(random.uniform(min_seconds, max_seconds))


def load_undetected_chromedriver():
    """Opsiyonel paketi dinamik yukler; yoksa None doner."""
    try:
        return importlib.import_module("undetected_chromedriver")
    except ImportError:
        return None


def get_chrome_major_version():
    """Mac'teki Chrome surumunun ana numarasini bulur: 147.0... -> 147."""
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        str(Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    ]

    for chrome_path in chrome_paths:
        if not Path(chrome_path).exists():
            continue

        try:
            result = subprocess.run(
                [chrome_path, "--version"],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue

        match = re.search(r"(\d+)\.", result.stdout)
        if match:
            return int(match.group(1))

    return None


def build_driver(use_undetected=USE_UNDETECTED_CHROME, headless=False):
    """Varsayilan olarak standart Selenium Chrome'u baslatir."""
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=tr-TR")
    options.add_argument("--remote-allow-origins=*")

    if headless:
        # Headless mod Google tarafinda daha hizli engellenebilir; gerekmedikce False tutun.
        options.add_argument("--headless=new")

    uc = load_undetected_chromedriver()
    if use_undetected and uc is not None:
        chrome_major_version = get_chrome_major_version()
        try:
            if chrome_major_version:
                print(f"Chrome surumu algilandi: {chrome_major_version}")
                return uc.Chrome(options=options, version_main=chrome_major_version)

            return uc.Chrome(options=options)
        except SessionNotCreatedException as exc:
            print(f"undetected-chromedriver baslatilamadi, standart Selenium deneniyor: {exc.msg}")
        except WebDriverException as exc:
            print(f"undetected-chromedriver hatasi, standart Selenium deneniyor: {exc.msg}")

    print("Standart Selenium Chrome baslatiliyor.")
    return Chrome(options=options)


def is_driver_alive(driver):
    """Chrome penceresi hala acik mi kontrol eder."""
    try:
        return bool(driver.window_handles)
    except WebDriverException:
        return False


def safe_click(driver, element):
    """Normal tiklama olmazsa JS tiklamasina duser."""
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
        human_sleep(0.2, 0.7)
        element.click()
    except (ElementClickInterceptedException, WebDriverException):
        driver.execute_script("arguments[0].click();", element)


def click_visible_element_by_keywords(driver, keywords, timeout=6):
    """
    Google Search ve Maps farkli etiketler kullanabiliyor. Bu fonksiyon gorunen
    tiklanabilir elementleri metin/aria-label ile bulup tiklar.
    """
    end_time = time.time() + timeout
    lowered_keywords = [keyword.lower() for keyword in keywords]

    while time.time() < end_time:
        try:
            element = driver.execute_script(
                """
                const keywords = arguments[0];
                const selector = [
                  'button',
                  'a',
                  '[role="button"]',
                  '[role="tab"]',
                  '[role="radio"]',
                  '[role="menuitem"]',
                  '[role="menuitemradio"]',
                  '[aria-haspopup]',
                  '[tabindex="0"]'
                ].join(',');

                const isVisible = (node) => {
                  const rect = node.getBoundingClientRect();
                  const style = window.getComputedStyle(node);
                  return rect.width > 0 && rect.height > 0 &&
                         style.visibility !== 'hidden' && style.display !== 'none';
                };

                for (const node of document.querySelectorAll(selector)) {
                  if (!isVisible(node)) continue;

                  const text = [
                    node.innerText,
                    node.textContent,
                    node.getAttribute('aria-label'),
                    node.getAttribute('data-value')
                  ].filter(Boolean).join(' ').toLowerCase();

                  if (keywords.some((keyword) => text.includes(keyword))) {
                    return node;
                  }
                }

                return null;
                """,
                lowered_keywords,
            )
            if element:
                safe_click(driver, element)
                human_sleep()
                return True
        except WebDriverException:
            pass

        human_sleep(0.4, 0.9)

    return False


def click_if_exists(driver, locators, timeout=4):
    """Verilen selector listesindeki ilk tiklanabilir elementi tiklar."""
    wait = WebDriverWait(driver, timeout)
    for by, selector in locators:
        try:
            element = wait.until(EC.element_to_be_clickable((by, selector)))
            safe_click(driver, element)
            human_sleep()
            return True
        except TimeoutException:
            continue
        except WebDriverException:
            continue
    return False


def accept_cookies_if_present(driver):
    """Google cerez/izin ekranlari cikarsa kabul etmeye calisir."""
    cookie_locators = [
        (By.CSS_SELECTOR, "button[aria-label*='Tümünü kabul et']"),
        (By.CSS_SELECTOR, "button[aria-label*='Accept all']"),
        (By.XPATH, "//button[.//*[contains(text(), 'Tümünü kabul et')] or contains(., 'Tümünü kabul et')]"),
        (By.XPATH, "//button[.//*[contains(text(), 'Accept all')] or contains(., 'Accept all')]"),
    ]
    click_if_exists(driver, cookie_locators, timeout=5)


def open_reviews_panel_if_needed(driver):
    """
    Link dogrudan yorumlar penceresini acmiyorsa 'Yorumlar' butonunu bulup tiklar.
    Mutlak XPath kullanilmaz; aria-label ve goreceli metin aramasi kullanilir.
    """
    review_button_locators = [
        (By.CSS_SELECTOR, "button[aria-label*='Yorumlar']"),
        (By.CSS_SELECTOR, "button[aria-label*='Reviews']"),
        (By.CSS_SELECTOR, "a[aria-label*='Yorumlar']"),
        (By.CSS_SELECTOR, "a[aria-label*='Reviews']"),
        (By.CSS_SELECTOR, "div[role='button'][aria-label*='Yorumlar']"),
        (By.CSS_SELECTOR, "div[role='button'][aria-label*='Reviews']"),
        (By.XPATH, "//*[self::button or self::a or @role='button' or @role='tab'][contains(., 'Google yorumları')]"),
        (By.XPATH, "//button[contains(., 'Yorumlar') or contains(., 'Reviews')]"),
        (By.XPATH, "//a[contains(@aria-label, 'Yorumlar') or contains(@aria-label, 'Reviews')]"),
        (By.XPATH, "//a[contains(., 'Yorumlar') or contains(., 'Reviews')]"),
    ]
    if click_if_exists(driver, review_button_locators, timeout=6):
        return True

    return click_visible_element_by_keywords(
        driver,
        ["yorumlar", "google yorumları", "reviews", "google reviews"],
        timeout=5,
    )


def find_scrollable_reviews_container(driver):
    """
    Google Search ve Maps farkli scroll panelleri kullanir. Yorum karti/puan
    iceren kaydirilabilir elementi puanlayarak secer.
    """
    try:
        container = driver.execute_script(
            """
            const isVisible = (node) => {
              const rect = node.getBoundingClientRect();
              const style = window.getComputedStyle(node);
              return rect.width > 200 && rect.height > 80 &&
                     style.visibility !== 'hidden' && style.display !== 'none';
            };

            let best = null;
            let bestScore = -1;
            const nodes = Array.from(document.querySelectorAll('div, main, section, [role="main"], [role="dialog"]'));

            for (const node of nodes) {
              if (!isVisible(node)) continue;
              if ((node.scrollHeight || 0) <= (node.clientHeight || 0) + 80) continue;

              const text = (node.innerText || node.textContent || '').slice(0, 2500);
              const reviewCards = node.querySelectorAll('div.bwb7ce[data-id], div[data-review-id], div.jftiEf').length;
              const ratings = node.querySelectorAll('[aria-label*="puan aldı"], [aria-label*="/5"], [aria-label*="yıldız"], [aria-label*="star"]').length;
              const hasReviewText = /Yorumlar|Reviews|Sıralama|Sort|puan aldı/i.test(text) ? 1 : 0;
              const scrollRoom = node.scrollHeight - node.clientHeight;
              const score = reviewCards * 1000 + ratings * 150 + hasReviewText * 500 + Math.min(scrollRoom, 3000);

              if (score > bestScore) {
                best = node;
                bestScore = score;
              }
            }

            return best;
            """
        )
        if container:
            return container
    except WebDriverException:
        pass

    candidates = []
    selectors = [
        "div.yFm7Pc",
        "div[role='dialog']",
        "div[role='feed']",
        "div[jsname]",
        "g-scrolling-carousel",
        "div[role='main'] div[tabindex='-1']",
        "div[role='main'] div[jslog]",
        "div[role='main'] div",
        "div[aria-label*='Yorumlar']",
        "div[aria-label*='Reviews']",
    ]

    for selector in selectors:
        try:
            candidates.extend(driver.find_elements(By.CSS_SELECTOR, selector))
        except WebDriverException:
            continue

    best_container = None
    best_scroll_height = 0

    for element in candidates:
        try:
            scroll_height = driver.execute_script("return arguments[0].scrollHeight || 0;", element)
            client_height = driver.execute_script("return arguments[0].clientHeight || 0;", element)
            if scroll_height > client_height and scroll_height > best_scroll_height:
                best_container = element
                best_scroll_height = scroll_height
        except (StaleElementReferenceException, WebDriverException):
            continue

    return best_container


def click_sort_option(driver, option_texts):
    """Siralama menusu icindeki hedef secenegi metinle bulur."""
    if click_visible_element_by_keywords(driver, option_texts, timeout=3):
        return True

    sort_button_locators = [
        (By.CSS_SELECTOR, "button[aria-label*='Sırala']"),
        (By.CSS_SELECTOR, "button[aria-label*='Sıralama']"),
        (By.CSS_SELECTOR, "button[aria-label*='Sort']"),
        (By.CSS_SELECTOR, "div[role='button'][aria-label*='Sırala']"),
        (By.CSS_SELECTOR, "div[role='button'][aria-label*='Sıralama']"),
        (By.CSS_SELECTOR, "div[role='button'][aria-label*='Sort']"),
        (By.CSS_SELECTOR, "[aria-haspopup='menu'][aria-label*='Sırala']"),
        (By.CSS_SELECTOR, "[aria-haspopup='menu'][aria-label*='Sort']"),
        (By.XPATH, "//button[contains(., 'Sırala') or contains(., 'Sort')]"),
        (By.XPATH, "//*[@role='button' and (contains(., 'Sırala') or contains(., 'Sort') or contains(., 'En alakalı') or contains(., 'Most relevant'))]"),
    ]

    if not click_if_exists(driver, sort_button_locators, timeout=8) and not click_visible_element_by_keywords(
        driver,
        ["sırala", "sıralama", "sort", "en alakalı", "most relevant"],
        timeout=5,
    ):
        print("Siralama butonu bulunamadi; mevcut siralama ile devam ediliyor.")
        return False

    option_xpath = " | ".join(
        [
            f"//*[@role='menuitemradio' or @role='menuitem' or @role='option' or @role='radio'][contains(., '{text}')]"
            for text in option_texts
        ]
    )

    try:
        option = WebDriverWait(driver, 8).until(EC.element_to_be_clickable((By.XPATH, option_xpath)))
        safe_click(driver, option)
        human_sleep(2, 4)
        return True
    except TimeoutException:
        if click_visible_element_by_keywords(driver, option_texts, timeout=5):
            return True

        print(f"Siralama secenegi bulunamadi: {option_texts}")
        return False


def set_lowest_rating_sort(driver):
    return click_sort_option(
        driver,
        [
            "En düşük puan",
            "En düşük puanlı",
            "En düşük",
            "Düşük",
            "En Düşük",
            "Lowest rating",
            "Lowest rated",
            "Lowest",
        ],
    )


def set_highest_rating_sort(driver):
    return click_sort_option(
        driver,
        [
            "En yüksek puan",
            "En yüksek puanlı",
            "En yüksek",
            "Yüksek",
            "En Yüksek",
            "Highest rating",
            "Highest rated",
            "Highest",
        ],
    )


def expand_all_visible_reviews(driver):
    """Gorunen tum 'Daha fazla' butonlarina basarak uzun yorumlari acar."""
    more_button_locators = [
        "button[aria-label*='Daha fazla']",
        "button[aria-label*='More']",
        "button[aria-label*='Read more']",
        "a[aria-label*='devamını okuyun']",
        "a[aria-label*='Devamını okuyun']",
        "a[role='button'].MtCSLb",
        "span[role='button'][aria-label*='Daha fazla']",
        "span[role='button'][aria-label*='More']",
    ]

    for selector in more_button_locators:
        try:
            buttons = driver.find_elements(By.CSS_SELECTOR, selector)
        except WebDriverException:
            continue

        for button in buttons:
            try:
                if button.is_displayed() and button.is_enabled():
                    safe_click(driver, button)
                    human_sleep(0.2, 0.8)
            except (StaleElementReferenceException, WebDriverException):
                continue


def parse_star_value(review_card):
    """Yorum kartindaki aria-label bilgisinden 1-5 yildiz degerini cikarir."""
    rating_selectors = [
        "span[role='img'][aria-label*='yıldız']",
        "span[role='img'][aria-label*='Yıldız']",
        "span[role='img'][aria-label*='star']",
        "div[role='img'][aria-label*='puan aldı']",
        "div[role='img'][aria-label*='/5']",
        "span[aria-label*='yıldız']",
        "span[aria-label*='Yıldız']",
        "span[aria-label*='star']",
        "div[aria-label*='puan aldı']",
        "div[aria-label*='/5']",
        "g-review-stars span[aria-label]",
    ]

    for selector in rating_selectors:
        try:
            rating_element = review_card.find_element(By.CSS_SELECTOR, selector)
            if rating_element is None:
                continue
            aria_label = rating_element.get_attribute("aria-label") or ""
            match = re.search(r"([1-5])", aria_label)
            if match:
                return int(match.group(1))
        except NoSuchElementException:
            continue
        except StaleElementReferenceException:
            return None

    return None


def parse_review_text(review_card):
    """Yorum metnini Google'in yaygin yorum text alanlarindan ceker."""
    text_selectors = [
        "div.OA1nbd",
        "span.wiI7pd",
        "div.MyEned span",
        "span[jsname='bN97Pc']",
        "div[lang] span",
        "span.review-full-text",
        "span.review-snippet",
        "div.Jtu6Td",
        "div.review-dialog-list div[lang]",
    ]

    for selector in text_selectors:
        try:
            element = review_card.find_element(By.CSS_SELECTOR, selector)
            text = element.text.strip()
            if text:
                return text
        except NoSuchElementException:
            continue
        except StaleElementReferenceException:
            return ""

    return ""


def get_visible_review_cards(driver):
    """Sayfadaki gorunur yorum kartlarini esnek selectorlarla dondurur."""
    card_selectors = [
        "div[data-review-id]",
        "div.jftiEf",
        "div[aria-label][data-review-id]",
        "div.gws-localreviews__google-review",
        "div.bwb7ce[data-id]",
        "div[jscontroller][data-review-id]",
        "div[jscontroller]:has(span[aria-label*='yıldız'])",
        "div[jscontroller]:has(span[aria-label*='star'])",
    ]

    cards = []
    seen_ids = set()

    for selector in card_selectors:
        try:
            for card in driver.find_elements(By.CSS_SELECTOR, selector):
                element_id = card.get_attribute("data-review-id") or card.id
                if element_id not in seen_ids:
                    cards.append(card)
                    seen_ids.add(element_id)
        except WebDriverException:
            continue

    return cards


def get_reviews_with_javascript(driver):
    """
    Fallback okuyucu: Google Search yerel panelinde class isimleri degistiginde,
    yildiz aria-label'larini bulup yakin yorum metnini JS ile cikarir.
    """
    try:
        return driver.execute_script(
            """
            const results = [];
            const seen = new Set();

            const addResult = (star, reviewText) => {
              if (!star || !reviewText) return;
              const key = `${star}::${reviewText}`;
              if (!seen.has(key)) {
                results.push({ Yildiz: Number(star), Yorum: reviewText });
                seen.add(key);
              }
            };

            const clean = (text) => (text || '')
              .replace(/\\s+/g, ' ')
              .replace(/Daha fazla|Read more|More|Diğer/gi, '')
              .trim();

            const badText = (text) => {
              const lower = text.toLocaleLowerCase('tr-TR');
              return !text ||
                text.length < 4 ||
                lower.includes('yerel rehber') ||
                lower.includes('local guide') ||
                lower.includes('yorum') && text.length < 20 ||
                lower.includes('photo') ||
                lower.includes('foto') ||
                lower.includes('paylaş') ||
                lower.includes('tepki eklemek') ||
                lower.includes('function()');
            };

            for (const card of document.querySelectorAll('div.bwb7ce[data-id]')) {
              const ratingNode = card.querySelector('[aria-label*="puan aldı"], [aria-label*="/5"]');
              const ratingLabel = ratingNode ? ratingNode.getAttribute('aria-label') || '' : '';
              const ratingMatch = ratingLabel.match(/[1-5]/);
              if (!ratingMatch) continue;

              const textNodes = Array.from(card.querySelectorAll('div.OA1nbd, span.wiI7pd, div.MyEned span, div[lang], span[lang]'));
              const reviewText = textNodes
                .map((node) => clean(node.innerText || node.textContent))
                .filter((text) => !badText(text))
                .sort((a, b) => b.length - a.length)[0];

              addResult(Number(ratingMatch[0]), reviewText);
            }

            const ratingNodes = Array.from(
              document.querySelectorAll('[aria-label*="yıldız"], [aria-label*="Yıldız"], [aria-label*="star"], [aria-label*="Star"], [aria-label*="puan aldı"], [aria-label*="/5"]')
            );

            const isVisible = (node) => {
              const rect = node.getBoundingClientRect();
              const style = window.getComputedStyle(node);
              return rect.width > 0 && rect.height > 0 &&
                     style.visibility !== 'hidden' && style.display !== 'none';
            };

            for (const ratingNode of ratingNodes) {
              if (!isVisible(ratingNode)) continue;

              const ratingLabel = ratingNode.getAttribute('aria-label') || '';
              const ratingMatch = ratingLabel.match(/[1-5]/);
              if (!ratingMatch) continue;

              let container = ratingNode;
              for (let i = 0; i < 8 && container.parentElement; i += 1) {
                container = container.parentElement;

                const textCandidates = Array.from(container.querySelectorAll(
                  'span.wiI7pd, div.MyEned span, span.review-full-text, span.review-snippet, div.Jtu6Td, div[lang], span[lang]'
                ));

                const texts = textCandidates
                  .map((node) => clean(node.innerText || node.textContent))
                  .filter((text) => !badText(text));

                const reviewText = texts.sort((a, b) => b.length - a.length)[0];
                if (reviewText) {
                  addResult(Number(ratingMatch[0]), reviewText);
                  break;
                }
              }
            }

            return results;
            """
        )
    except WebDriverException:
        return []


def should_continue_collecting(collected_by_star, target_stars):
    return any(len(collected_by_star[star]) < TARGET_PER_STAR for star in target_stars)


def get_unfinished_star(collected_by_star, target_stars, skipped_stars=None):
    """Hedef siralama icinde hala toplanmasi gereken ilk yildizi dondurur."""
    skipped_stars = skipped_stars or set()
    for star in target_stars:
        if star in skipped_stars:
            continue
        if len(collected_by_star[star]) < TARGET_PER_STAR:
            return star
    return None


def is_element_in_viewport(driver, element):
    """Element su an ekranda gorunur mu kontrol eder."""
    try:
        return bool(
            driver.execute_script(
                """
                const element = arguments[0];
                const rect = element.getBoundingClientRect();
                const style = window.getComputedStyle(element);
                return rect.width > 0 && rect.height > 0 &&
                       rect.bottom > 0 && rect.top < window.innerHeight &&
                       style.visibility !== 'hidden' && style.display !== 'none';
                """,
                element,
            )
        )
    except (StaleElementReferenceException, WebDriverException):
        return False


def get_visible_review_stars(driver):
    """Ekranda gorunen yorum kartlarindaki yildiz degerlerini okur."""
    visible_stars = []

    for card in get_visible_review_cards(driver):
        if not is_element_in_viewport(driver, card):
            continue

        try:
            star = parse_star_value(card)
            if star in STAR_VALUES:
                visible_stars.append(star)
        except (StaleElementReferenceException, WebDriverException):
            continue

    if visible_stars:
        return visible_stars

    try:
        visible_stars = driver.execute_script(
            """
            const isVisible = (node) => {
              const rect = node.getBoundingClientRect();
              const style = window.getComputedStyle(node);
              return rect.width > 0 && rect.height > 0 &&
                     rect.bottom > 0 && rect.top < window.innerHeight &&
                     style.visibility !== 'hidden' && style.display !== 'none';
            };

            return Array.from(document.querySelectorAll('div.bwb7ce[data-id], div[data-review-id], div.jftiEf'))
              .filter(isVisible)
              .map((card) => {
                const ratingNode = card.querySelector('[aria-label*="puan aldı"], [aria-label*="/5"], [aria-label*="yıldız"], [aria-label*="star"]');
                const label = ratingNode ? ratingNode.getAttribute('aria-label') || '' : '';
                const match = label.match(/[1-5]/);
                return match ? Number(match[0]) : null;
              })
              .filter(Boolean);
            """
        )
    except WebDriverException:
        visible_stars = []

    return visible_stars


def add_visible_reviews(driver, collected_by_star, target_stars):
    """Gorunen yorumlari hedef yildiz kovalarina ekler."""
    expand_all_visible_reviews(driver)

    added_count = 0
    for card in get_visible_review_cards(driver):
        try:
            star = parse_star_value(card)
            if star not in target_stars:
                continue
            if len(collected_by_star[star]) >= TARGET_PER_STAR:
                continue

            review_text = parse_review_text(card)
            if not review_text:
                continue

            # Ayni yildiz kategorisinde tekrar eden yorumlari engelle.
            if review_text in collected_by_star[star]:
                continue

            collected_by_star[star].append(review_text)
            added_count += 1
            print(f"{star} yildiz yorumu alindi: {len(collected_by_star[star])}/{TARGET_PER_STAR}")
        except (StaleElementReferenceException, WebDriverException) as exc:
            print(f"Yorum karti okunamadi, atlandi: {exc}")

    if added_count > 0:
        return added_count

    for review in get_reviews_with_javascript(driver):
        star = review.get("Yildiz")
        review_text = (review.get("Yorum") or "").strip()
        if star not in target_stars:
            continue
        if len(collected_by_star[star]) >= TARGET_PER_STAR:
            continue
        if not review_text or review_text in collected_by_star[star]:
            continue

        collected_by_star[star].append(review_text)
        added_count += 1
        print(f"{star} yildiz yorumu alindi JS: {len(collected_by_star[star])}/{TARGET_PER_STAR}")

    return added_count


def scroll_reviews(driver, scroll_container):
    """Yorum panelini asagi kaydirir; container yoksa klavye ile fallback yapar."""
    moved = False

    try:
        if scroll_container:
            moved = bool(
                driver.execute_script(
                    """
                    const element = arguments[0];
                    const before = element.scrollTop;
                    const amount = Math.max(element.clientHeight * 0.85, 650);
                    element.scrollTop = Math.min(element.scrollTop + amount, element.scrollHeight);
                    return element.scrollTop !== before;
                    """,
                    scroll_container,
                )
            )
        else:
            moved = bool(
                driver.execute_script(
                    """
                    const before = window.scrollY;
                    window.scrollBy(0, Math.max(window.innerHeight * 0.85, 650));
                    return window.scrollY !== before;
                    """
                )
            )
    except (StaleElementReferenceException, WebDriverException):
        try:
            before_position = driver.execute_script("return window.scrollY;")
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
            moved = driver.execute_script("return window.scrollY;") != before_position
        except WebDriverException:
            moved = False

    human_sleep(1, 3)
    return moved


def dump_debug_snapshot(driver, label):
    """Veri bulunamazsa sayfanin HTML ve screenshot halini debug klasorune yazar."""
    if not is_driver_alive(driver):
        print("Chrome penceresi kapali oldugu icin debug kaydi alinamadi.")
        return

    DEBUG_DIR.mkdir(exist_ok=True)
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_")[:80] or "page"
    timestamp = int(time.time())
    html_path = DEBUG_DIR / f"{timestamp}_{safe_label}.html"
    screenshot_path = DEBUG_DIR / f"{timestamp}_{safe_label}.png"

    try:
        html_path.write_text(driver.page_source, encoding="utf-8")
        driver.save_screenshot(str(screenshot_path))
        print(f"Debug kaydi olusturuldu: {html_path.name}, {screenshot_path.name}")
    except WebDriverException as exc:
        print(f"Debug kaydi olusturulamadi: {exc}")


def collect_reviews_for_stars(driver, collected_by_star, target_stars, max_scrolls=120):
    """
    Istenen yildizlar icin sirali sekilde kaydirir.
    1 yildiz dolduktan sonra 2 yildiza, 5 yildiz dolduktan sonra 4 yildiza
    ulasana kadar erken durmaz.
    """
    scroll_container = find_scrollable_reviews_container(driver)
    skipped_stars = set()
    passed_star_scrolls = {star: 0 for star in target_stars}
    no_movement_scrolls = 0
    ascending_order = target_stars == sorted(target_stars)

    for scroll_index in range(max_scrolls):
        add_visible_reviews(driver, collected_by_star, target_stars)

        current_target = get_unfinished_star(collected_by_star, target_stars, skipped_stars)
        if current_target is None:
            break

        visible_stars = get_visible_review_stars(driver)
        if visible_stars:
            visible_summary = ", ".join(str(star) for star in sorted(set(visible_stars)))
            print(f"Gorunen yildizlar: {visible_summary}; aranan: {current_target}")

            if ascending_order:
                passed_current_target = min(visible_stars) > current_target
            else:
                passed_current_target = max(visible_stars) < current_target

            if passed_current_target:
                passed_star_scrolls[current_target] += 1
            else:
                passed_star_scrolls[current_target] = 0

            if passed_star_scrolls[current_target] >= 4:
                skipped_stars.add(current_target)
                print(
                    f"{current_target} yildiz yorumu gorunmedi; "
                    "siralama sonraki yildizlara gecti, mevcut olanlarla devam ediliyor."
                )
                continue
        else:
            print(f"Gorunen yorum yildizi okunamadi; {current_target} yildiz icin kaydiriliyor.")

        moved = scroll_reviews(driver, scroll_container)
        if moved:
            no_movement_scrolls = 0
        else:
            no_movement_scrolls += 1

        if no_movement_scrolls >= 8:
            print("Sayfa daha fazla kaymiyor; bu siralama icin mevcut yorumlarla devam ediliyor.")
            break

        if (scroll_index + 1) % 15 == 0:
            remaining = [
                f"{star}:{len(collected_by_star[star])}/{TARGET_PER_STAR}"
                for star in target_stars
                if star not in skipped_stars
            ]
            print("Toplama durumu: " + ", ".join(remaining))


def scrape_single_business(driver, review_url):
    """Tek bir Google Isletme/Yorum linkinden 1-5 yildizli yorumlari toplar."""
    collected_by_star = {star: [] for star in STAR_VALUES}

    try:
        if not is_driver_alive(driver):
            print("Chrome penceresi kapali; bu link islenemedi.")
            return []

        print(f"\nLink aciliyor: {review_url}")
        driver.get(review_url)
        human_sleep(4, 7)
        accept_cookies_if_present(driver)
        open_reviews_panel_if_needed(driver)
        human_sleep(2, 4)

        print("En dusuk puanli siralama ile 1, 2, 3 ve 4 yildizli yorumlar toplaniyor.")
        set_lowest_rating_sort(driver)
        collect_reviews_for_stars(driver, collected_by_star, LOW_SORT_STAR_VALUES)

        print("En yuksek puanli siralama ile 5 yildizli ve eksik kalan 4 yildizli yorumlar toplaniyor.")
        set_highest_rating_sort(driver)
        collect_reviews_for_stars(driver, collected_by_star, HIGH_STAR_VALUES)

    except TimeoutException as exc:
        print(f"Sayfa zaman asimina ugradi, link atlandi: {exc}")
    except WebDriverException as exc:
        print(f"WebDriver hatasi, link atlandi: {exc}")
    except Exception as exc:
        print(f"Beklenmeyen hata, link atlandi: {exc}")

    rows = []
    for star in STAR_VALUES:
        for review in collected_by_star[star]:
            rows.append({"Yorum": review, "Yildiz": star})

    if not rows:
        dump_debug_snapshot(driver, f"empty_reviews_{abs(hash(review_url))}")

    return rows


def save_dataset(rows):
    """Yorumlari cift tirnakli CSV olarak UTF-8 BOM ile kaydeder."""
    df = pd.DataFrame(rows, columns=["Yorum", "Yildiz"])
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
    )
    print(f"\nKaydedildi: {OUTPUT_FILE} ({len(df)} satir)")


def main():
    if not GOOGLE_REVIEW_LINKS:
        print("GOOGLE_REVIEW_LINKS listesine en az bir Google yorum linki ekleyin.")
        return

    all_rows = []
    driver = build_driver(headless=False)

    try:
        for review_url in GOOGLE_REVIEW_LINKS:
            if not is_driver_alive(driver):
                print("Chrome penceresi kapandi; yeni Chrome oturumu aciliyor.")
                driver = build_driver(headless=False)

            business_rows = scrape_single_business(driver, review_url)
            all_rows.extend(business_rows)
            save_dataset(all_rows)
            human_sleep(3, 6)
    finally:
        if is_driver_alive(driver):
            driver.quit()

    save_dataset(all_rows)


if __name__ == "__main__":
    main()
