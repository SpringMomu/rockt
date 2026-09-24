"""Repeatable visual QA: real scenario snapshot, stationary camera, evolving CFD."""
import json
from pathlib import Path
from urllib.request import Request, urlopen
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8766'
req = Request(URL + '/api/input', data=json.dumps({'actions':['scenario:3','pause']}).encode(), headers={'Content-Type':'application/json'})
urlopen(req).read()
state = json.load(urlopen(URL + '/api/state'))
state['paused'] = False
state['cfd_forces'] = False
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe', headless=True,
        args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--disable-web-security'])
    page = browser.new_page(viewport={'width':1600,'height':1000}, device_scale_factor=1)
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('console', lambda m: errors.append(m.text) if m.type=='error' else None)
    page.route('**/api/state', lambda r: r.fulfill(json=state))
    page.route('**/api/cfd3d', lambda r: r.fulfill(json={'ok':True}))
    page.goto(URL+'/?cfd=low')
    page.wait_for_function('window.__app?.fluid?.steps > 130', timeout=180000)
    page.screenshot(path=str(ROOT/'airflow-default.png'))
    print(json.dumps(page.evaluate('({steps:__app.fluid.steps,ok:__app.fluid.ok,glError:__app.fluid.gl.getError(),renderer:__app.fluid.gl.getParameter(__app.fluid.gl.RENDERER)})')),flush=True)
    page.keyboard.press('b')
    page.wait_for_function('(start) => __app.fluid.steps > start + 9', arg=page.evaluate('__app.fluid.steps'), timeout=30000)
    page.screenshot(path=str(ROOT/'airflow-vorticity.png'))
    page.keyboard.press('b')
    page.wait_for_function('(start) => __app.fluid.steps > start + 9', arg=page.evaluate('__app.fluid.steps'), timeout=30000)
    page.screenshot(path=str(ROOT/'airflow-speed.png'))
    page.keyboard.press('b')
    page.wait_for_function('(start) => __app.fluid.steps > start + 9', arg=page.evaluate('__app.fluid.steps'), timeout=30000)
    page.screenshot(path=str(ROOT/'airflow-dye.png'))
    print(json.dumps({'errors':errors},ensure_ascii=False),flush=True)
    (ROOT/'airflow-browser-check.json').write_text(json.dumps({'errors':errors},indent=2),encoding='utf-8')
    browser.close()
    assert not errors

