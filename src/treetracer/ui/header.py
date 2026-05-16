"""Top header bar: logo, title, sidebar toggle, log toggle, dark-mode toggle."""

import dash_mantine_components as dmc

from ..icons import icon


def add_header():
    logo_path = "assets/treetracer-icon.png"
    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Tooltip(
                    dmc.ActionIcon(
                        icon("tabler:layout-sidebar-left-collapse", size=28),
                        id="sidebar-toggle",
                        variant="subtle",
                        size="xl",
                    ),
                    label="Toggle Sidebar",
                ),
                dmc.Image(src=logo_path, w=50, fit="contain"),
                dmc.Title("TreeTracer", c="blue"),
                dmc.Space(style={"flex": 1}),
                dmc.Tooltip(
                    dmc.ActionIcon(
                        icon("tabler:info-circle", size=20),
                        id="about-modal-button",
                        variant="subtle",
                        size="lg",
                    ),
                    label="About",
                ),
                dmc.Tooltip(
                    dmc.ActionIcon(
                        icon("tabler:terminal-2", size=20),
                        id="log-toggle-button",
                        variant="subtle",
                        size="lg",
                    ),
                    label="Toggle Log",
                ),
                dmc.Tooltip(
                    dmc.ActionIcon(
                        icon("tabler:moon", size=20, id="dark-mode-icon"),
                        id="dark-mode-toggle",
                        variant="subtle",
                        size="lg",
                    ),
                    label="Toggle dark mode",
                ),
            ],
            h="100%",
            px="md",
            gap="xs",
            wrap="nowrap",
        )
    )
