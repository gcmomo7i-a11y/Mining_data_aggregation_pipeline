"""工具函数单元测试."""

from pipeline.utils import HTMLCleaner, compute_dedup_hash


def test_html_cleaner_basic() -> None:
    """测试HTMLCleaner基本清洗."""
    cleaner = HTMLCleaner()
    html = "<p>Hello World</p>"
    result = cleaner.clean(html)
    assert "Hello World" in result


def test_html_cleaner_removes_script() -> None:
    """测试HTMLCleaner移除script标签."""
    cleaner = HTMLCleaner()
    html = "<script>alert('xss')</script><p>Content</p>"
    result = cleaner.clean(html)
    assert "alert" not in result
    assert "Content" in result


def test_html_cleaner_removes_style() -> None:
    """测试HTMLCleaner移除style标签."""
    cleaner = HTMLCleaner()
    html = "<style>body{color:red;}</style><p>Content</p>"
    result = cleaner.clean(html)
    assert "color" not in result
    assert "Content" in result


def test_html_cleaner_removes_nav_footer() -> None:
    """测试HTMLCleaner移除nav/footer/header标签."""
    cleaner = HTMLCleaner()
    html = "<nav>导航</nav><main>正文</main><footer>页脚</footer>"
    result = cleaner.clean(html)
    assert "导航" not in result
    assert "页脚" not in result
    assert "正文" in result


def test_html_cleaner_extract_main_content() -> None:
    """测试HTMLCleaner提取主要内容区域."""
    cleaner = HTMLCleaner()
    html = "<nav>Nav</nav><div class='content'>Main Content</div><footer>Footer</footer>"
    result = cleaner.extract_main_content(html, "div.content")
    assert "Main Content" in result
    assert "Nav" not in result


def test_html_cleaner_normalize_whitespace() -> None:
    """测试HTMLCleaner规范化空白."""
    cleaner = HTMLCleaner()
    html = "<p>Line1</p><p>Line2</p><p>Line3</p>"
    result = cleaner.clean(html)
    assert "Line1" in result
    assert "Line2" in result


def test_dedup_hash_deterministic() -> None:
    """测试去重哈希确定性."""
    hash1 = compute_dedup_hash("标题", "2026-07-01")
    hash2 = compute_dedup_hash("标题", "2026-07-01")
    assert hash1 == hash2


def test_dedup_hash_different_inputs() -> None:
    """测试不同输入产生不同哈希."""
    hash1 = compute_dedup_hash("标题A", "2026-07-01")
    hash2 = compute_dedup_hash("标题B", "2026-07-01")
    assert hash1 != hash2


def test_dedup_hash_whitespace_normalization() -> None:
    """测试空白字符不影响哈希."""
    hash1 = compute_dedup_hash("标题", "2026-07-01")
    hash2 = compute_dedup_hash(" 标题 ", " 2026-07-01 ")
    assert hash1 == hash2
