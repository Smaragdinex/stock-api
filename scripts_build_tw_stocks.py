from pathlib import Path
import json
import re
import subprocess

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / 'tw_stocks.json'
CACHE_DIR = BASE_DIR / '.cache'
CACHE_DIR.mkdir(exist_ok=True)

SOURCES = [
    {
        'url': 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2',
        'exchange': 'TWSE',
        'market': '上市',
        'suffix': '.TW',
        'cache': CACHE_DIR / 'twse.html',
    },
    {
        'url': 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=4',
        'exchange': 'TWO',
        'market': '上櫃',
        'suffix': '.TWO',
        'cache': CACHE_DIR / 'tpex.html',
    },
]

CODE_NAME_RE = re.compile(r'^(\d{4,6})[\s\u3000]+(.+)$')


def fetch_table(source: dict):
    cache_path = source['cache']
    subprocess.run(['curl', '-L', source['url'], '-o', str(cache_path)], check=True)
    return pd.read_html(str(cache_path), encoding='cp950')[0]


def parse_equities(df: pd.DataFrame, exchange: str, market: str, suffix: str):
    results = []

    for _, row in df.iterrows():
        code_name = str(row.iloc[0]).strip()
        market_value = str(row.iloc[3]).strip() if len(row) > 3 else ''
        cfi_code = str(row.iloc[5]).strip() if len(row) > 5 else ''

        match = CODE_NAME_RE.match(code_name)
        if not match:
            continue

        code, name = match.groups()
        name = name.strip()

        if market_value != market:
            continue
        if not cfi_code.startswith('ES') and not cfi_code.startswith('CE'):
            continue

        quote_type = 'ETF' if 'ETF' in name.upper() or cfi_code.startswith('CE') else 'EQUITY'
        results.append({
            'symbol': f'{code}{suffix}',
            'code': code,
            'name': name,
            'exchange': exchange,
            'market': market,
            'type': quote_type,
        })

    return results


def main():
    all_items = []
    for source in SOURCES:
        df = fetch_table(source)
        all_items.extend(parse_equities(df, source['exchange'], source['market'], source['suffix']))

    deduped = {}
    for item in all_items:
        deduped[item['symbol']] = item

    ordered = sorted(deduped.values(), key=lambda x: (x['code'], x['symbol']))
    OUT_PATH.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {len(ordered)} symbols to {OUT_PATH}')


if __name__ == '__main__':
    main()
