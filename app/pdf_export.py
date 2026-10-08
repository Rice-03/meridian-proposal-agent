"""Make the proposal PDF on the server, for the WhatsApp reply.

The PDF is normally made inside the browser: the generator's "Export PDF" button draws each page
with html2canvas and assembles them with jsPDF. So to get the same file on the server, this opens
the generator in a hidden Chrome (Playwright), loads the proposal with window.loadProposal(), presses
the same button and keeps the file that is downloaded. Nothing about the generator is changed.

Playwright is only needed for this one feature and is imported inside the function, so the rest of
the app works without it. One-time install:

    pip install playwright
    python -m playwright install chromium
"""

PAGE_PATH = "/generator/challenge-generator.html"


class PdfExportError(Exception):
    """Raised with a short, readable reason when the PDF could not be made."""


def export_pdf(proposal, base_url, dest, timeout_seconds=150, prepare=None):
    """Write the generator's PDF for `proposal` to `dest` (a path). `base_url` is where this app is
    running, for example http://127.0.0.1:8000. `prepare(page)` is a hook for tests only."""
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise PdfExportError("Playwright is not installed (pip install playwright, then: python -m playwright install chromium).") from exc

    ms = int(timeout_seconds * 1000)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page(accept_downloads=True, viewport={"width": 1280, "height": 1000})
                if prepare:
                    prepare(page)
                page.goto(base_url.rstrip("/") + PAGE_PATH, wait_until="load", timeout=ms)
                # the generator needs a moment before it accepts a proposal ("App not ready")
                page.wait_for_function("typeof window.loadProposal === 'function'", timeout=ms)
                page.evaluate(
                    """async (p) => {
                        for (let i = 0; i < 100; i++) {
                            try { window.loadProposal(p, { preview: true }); return true; }
                            catch (e) { if (!/not ready/i.test(String(e && e.message))) throw e; }
                            await new Promise(r => setTimeout(r, 100));
                        }
                        throw new Error('The generator did not become ready.');
                    }""",
                    proposal,
                )
                page.wait_for_selector(".pdf-doc .page", state="attached", timeout=ms)
                with page.expect_download(timeout=ms) as download:
                    page.evaluate("exportPdf()")
                download.value.save_as(str(dest))
            finally:
                browser.close()
    except PlaywrightTimeout as exc:
        raise PdfExportError("Making the PDF took too long. The generator loads two libraries from the internet; check the connection.") from exc
    except PlaywrightError as exc:
        raise PdfExportError("Could not make the PDF: " + str(exc).splitlines()[0][:200]) from exc
    return dest