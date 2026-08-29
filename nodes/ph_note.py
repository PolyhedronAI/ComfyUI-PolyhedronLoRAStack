"""
ph_note.py -- the Polyhedron Note.

WHY A NOTE AT ALL, when ComfyUI already has one: Core's `Note` is a FRONTEND
node. It has no Python class -- checked at Frank's own Core (v0.33.4) and at
master: `class Note` is nowhere in nodes.py. Ours exists so the suite owns the
node it ships, and so a note can carry a colour that survives a save.

WHAT THIS NODE IS, after v892: a coloured card with a heading and a body. That
is the whole of it. Frank's three notes in the MiniMax H3 workflow are switch
legends --

    1 = NO UPSCALE            1 = NO VIDEO INTERPOLATION
    2 = UPSCALE               2 = VIDEO INTERPOLATION

-- and v890 tried to LINK such a note to its switch and highlight the live
line. That went again in v892, for reasons that were measured, not argued:
the click that picked the target was hooked onto a LiteGraph method carrying
`@deprecated` which nothing ever calls, so the link could not be made on any
frontend in use; and in Frank's own graph each note already sits ~130 px
directly above its switch, which shows its number in plain sight. The feature
bought one short glance and charged an arming mode, a second click and a node
id that dangles as soon as the node is copied.

The promise it advertised was also never kept: it matched NUMBERS, never
meaning. A note reading "2 = UPSCALE" whose input 2 bypasses the upscaler was
highlighted with the same confidence. Retired rather than weakened.

NO INPUTS, NO OUTPUTS -- deliberately. A node with no outputs and no
OUTPUT_NODE flag is never part of any execution path, so ComfyUI never runs it:
the note costs nothing at queue time and cannot fail a run. `noop` exists only
because ComfyUI wants a FUNCTION to point at; it is never called. Same shape as
ULSEverywhere, which has carried RETURN_TYPES = () since v573.

WHERE THE COLOUR LIVES, after v893: nowhere of ours. The row on the node
drives ComfyUI's OWN colour set -- the nine of LGraphCanvas.node_colors plus
"none" -- through `setColorOption` / `getColorOption`, the very methods the
right-click > Colors menu uses. So the row and that menu are one system and
cannot disagree, and the state sits in LiteGraph's `color`/`bgcolor` where it
already belonged.

v892 briefly kept a colour of its own in node.properties and derived a body
tint from it. That was withdrawn: to stop the derived value being written
twice, its serialize() deleted `color`/`bgcolor` -- and thereby also deleted
any colour set through the right-click menu, silently, visible only after a
reload. A note saved by v892 is translated once on load and the property is
removed.

Either way the backend never reads any of it. Nothing about the colour is a
widget: appending one would have cost a widget index in every saved workflow
(guard #577) for a value Python never looks at. That is the v750 decision,
unchanged.
"""

MAX_TITLE = 200


class ULSNote:
    """A coloured card that explains the graph around it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "title": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Heading, drawn bold above the text. Leave it "
                               "empty for a plain note.",
                }),
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "The note itself. Free text -- write whatever "
                               "the graph around it needs explained.",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "Polyhedron/Logic"
    DESCRIPTION = (
        "A coloured note that belongs to the graph. Write a heading and a "
        "body; the row on the node sets the same colours as right-click > "
        "Colors, in one click, and folds away when you do not want it. It "
        "has no inputs and no outputs: it is never executed and costs "
        "nothing when you queue a prompt."
    )

    def noop(self):
        # Never called: with no outputs this node is not in any execution path.
        return ()
