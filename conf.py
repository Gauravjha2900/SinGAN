# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'SinGAN'
copyright = '2021, xinetzone'
author = 'xinetzone'

# The full version, including alpha/beta/rc tags
release = '0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_book_theme",
    "myst_nb",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx_thebe",
    "sphinx_copybutton",
    "sphinx_comments",
    # "sphinx.ext.todo",
    # "sphinxcontrib.bibtex",
    # "sphinx_togglebutton",
    # "sphinx.ext.viewcode",
    # "sphinx.ext.autodoc",
    # "sphinx.ext.doctest",
    # "sphinx_design",
    # "sphinx.ext.ifconfig",
    # "sphinx_automodapi.automodapi",
    # "sphinxext.opengraph",
]

myst_enable_extensions = [
    "colon_fence",
    "amsmath",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
    # "linkify",
]

comments_config = {
    "hypothesis": True,
    "dokieli": False,
    "utterances": {
        "repo": "daobook/SinGAN",
        "optional": "config",
    }
}

# MyST NB 设置
nb_render_priority = {
    "html": (
        "application/vnd.jupyter.widget-view+json",
        "application/javascript",
        "text/html",
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "text/markdown",
        "text/latex",
        "text/plain",
    ),
    'gettext': ()
}

extlinks = {
    'github': ('https://github.com/%s', ''),
    'daobook': ('https://daobook.github.io/%s', ''),
}

# intersphinx_mapping = {
#     'python': ('https://daobook.github.io/cpython/', None)
# }

# Add any paths that contain templates here, relative to this directory.
templates_path = ['docs/_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'zh_CN'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['docs/_static']
html_css_files = ["default.css"]

# -- 国际化输出 ----------------------------------------------------------------

locale_dirs = ['docs/locales/']  # path is example but recommended.
gettext_compact = False  # optional.

# -- 主题设置 -------------------------------------------------------------------

# 定制主侧栏
html_sidebars = {
    "*": [
        # 显示标志和网站标题。
        "sidebar-logo.html",
        #一个基于 bootstrap 的搜索栏（来自 PyData Sphinx Theme）
        "search-field.html",
        # 一个用于你的书基于 bootstrap 的导航菜单。
        "sbt-sidebar-nav.html",
        # 一个 可配置的 HTML 片段，用于添加到侧边栏（默认情况下，它被放置在底部）。
        "sbt-sidebar-footer.html",
    ],
    "posts/**": [
        "postcard.html",
        "recentposts.html",
        "tagcloud.html",
        "categories.html",
        "archives.html",
    ],
}

extra_navbar = """<div>
版权所有 © 2021 <a href="https://xinetzone.github.io/">xinetzone</a></div>
<div>由 <a href="https://ebp.jupyterbook.org/">EBP</a> 提供技术支持</div>
"""

html_theme_options = {
    # -- 如果你的文档只有一个页面，而且你不需要左边的导航栏，那么 ---------------
    # 你可以在 单页模式 下运行，
    # "single_page": False,  # 默认 `False`
    # 默认情况下，编辑按钮将指向版本库的根。
    # 如果你的文档被托管在一个子文件夹中，请使用以下配置：
    "path_to_docs": "docs/",  # 文档的路径，默认 `docs/``
    "repository_url": "https://github.com/daobook/SinGAN",
    "repository_branch": "main",  # 文档库的分支，默认 `master`
    # -- 在导航栏添加一个按钮，链接到版本库的议题 ------------------------------
    # （与 `repository_url` 和 `repository_branch` 一起使用）
    "use_issues_button": True,  # 默认 `False`
    # -- 在导航栏添加一个按钮，以下载页面的源文件。
    "use_download_button": True,  # 默认 `True`
    # 你可以在每个页面添加一个按钮，允许用户直接编辑页面文本，
    # 并提交拉动请求以更新文档。
    "use_edit_page_button": True,
    # 在导航栏添加一个按钮来切换全屏的模式。
    "use_fullscreen_button": True,  # 默认 `True`
    # -- 在导航栏中添加一个链接到文档库的按钮。----------------------------------
    "use_repository_button": True,  # 默认 `False`
    # -- 包含从 Jupyter 笔记本建立页面的 Binder 启动按钮。 ---------------------
    # "launch_buttons": '', # 默认 `False`
    "home_page_in_toc": False,  # 是否将主页放在导航栏（顶部）
    # -- 只显示标识，不显示 `html_title`，如果它存在的话。-----
    # "logo_only": True,
    # -- 在导航栏中显示子目录，向下到这里列出的深度。 ----
    # "show_navbar_depth": 2,
    # -- 在侧边栏页脚添加额外的 HTML -------------------
    # （如果 `sbt-sidebar-footer.html `在 `html_sidebars` 中被使用）。
    "extra_navbar": extra_navbar,
    # -- 在每个页面的页脚添加额外的 HTML。---
    # "extra_footer": '',
    # （仅限开发人员）触发一些功能，使开发主题更容易。
    # "theme_dev_mode": False
    # 重命名页内目录名称
    "toc_title": "导航",
    "launch_buttons": {
        # https://mybinder.org/v2/gh/daobook/SinGAN/main
        "binderhub_url": "https://mybinder.org",
        # "jupyterhub_url": "https://datahub.berkeley.edu",  # For testing
        "colab_url": "https://colab.research.google.com/",
        # 你可以控制有人点击启动按钮时打开的界面。
        "notebook_interface": "jupyterlab",
        "thebe": True,  # Thebe 实时代码单元格
    },
}
# -- 自定义网站的标志 --------------
html_logo = 'docs/logo.jpg'
# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "docs/page-logo.jfif"

# 如果你希望stderr和stdout中的每个输出都被合并成一个流，请使用以下配置。
# 避免将 jupter 执行报错的信息输出到 cmd
nb_merge_streams = True
execution_allow_errors = True
jupyter_execute_notebooks = 'off' # "cache"

epub_show_urls = 'footnote'