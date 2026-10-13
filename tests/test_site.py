"""Opt-in real-browser tests of the qualitative site, not the physics engine."""

import importlib.util
import os
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
ENABLED = (
    os.environ.get("BHID_BROWSER_TESTS") == "1"
    and importlib.util.find_spec("playwright") is not None
)


@unittest.skipUnless(ENABLED, "Opt-in real browser verification: BHID_BROWSER_TESTS=1")
class SiteBrowserTests(unittest.TestCase):
    def setUp(self):
        from playwright.sync_api import sync_playwright

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(ROOT)))
        self.addCleanup(self.server.server_close)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join)
        self.addCleanup(self.server.shutdown)
        self.playwright = sync_playwright().start()
        self.addCleanup(self.playwright.stop)
        channel = os.environ.get("BHID_TEST_BROWSER", "chromium")
        self.browser = self.playwright.chromium.launch(
            **({"channel": channel} if channel != "chromium" else {})
        )
        self.addCleanup(self.browser.close)
        self.context = self.browser.new_context(
            reduced_motion="reduce", viewport={"width": 1365, "height": 1000}
        )
        self.addCleanup(self.context.close)
        self.context.route(
            "**/*",
            lambda route: (
                route.continue_()
                if urlsplit(route.request.url).hostname == "127.0.0.1"
                else route.abort()
            ),
        )
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(f"http://127.0.0.1:{self.server.server_port}/index.html")
        self.page.wait_for_function(
            "document.querySelector('#massBMetric').textContent === '4.0000'"
        )

    def change(self, element, value):
        self.page.locator(element).evaluate(
            "(node,value)=>{node.value=value;node.dispatchEvent(new Event('input',{bubbles:true}))}",
            str(value),
        )

    def test_browser_scores_are_illustrative_and_report_is_historical(self):
        self.change("#timeSlider", 900)
        self.assertIn(
            "Their sign supplies no evidence", self.page.locator("#findingText").inner_text()
        )
        self.assertIn(
            "not a measured information residual", self.page.locator(".info-caption").inner_text()
        )
        self.assertIn(
            "version 0.2.0 numerical audit", self.page.locator(".report-card").inner_text()
        )
        self.assertEqual(self.errors, [])
        if os.environ.get("BHID_SCREENSHOT"):
            path = Path(os.environ["BHID_SCREENSHOT"])
            path.parent.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(path), full_page=True)

    def test_diffusion_cannot_restart_after_the_illustration_hits_the_cutoff(self):
        for element, value in (
            ("#m0Control", 2),
            ("#alphaControl", 0.8),
            ("#sigmaControl", 0.15),
            ("#floorControl", 100),
        ):
            self.change(element, value)
        for time in (600, 700, 800, 900, 1000):
            self.change("#timeSlider", time)
            self.assertEqual(self.page.locator("#massBMetric").inner_text(), "1.0000")
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
