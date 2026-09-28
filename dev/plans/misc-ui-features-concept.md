# Misc minor UI features

- add hotkeys for undo (ctrl+z) and redo (ctrl+y) for the current mode (mask/draw)
- brush/mask/zoom hotkeys (underline hotkey character in button text, these hotkeys should only trigger while not focusing an input/textarea/dropdown)
    c clone
    s smudge
    d draw
    m mask
    p crop
    n expand
    k shrink
    r clear
    f fit
- ctrl+mouse wheel = change brush size, shift+mouse wheel = change opacity
- add button to toggle lora on/off without removing it (same size as the `-` button, there's a class for these small buttons)
    - dropdowns for disabled items should use the disabled ui state (reduced opactity)
    - button should be positioned between the dropdown and the `-` button (reduce dropdown size to make it fit)
    - take a screenshot and check the button aligns nicely with the edges of the nearby elements
    - disabled-state should be recorded in the saved png metadata and be restored correctly on import
- also implement the on/off toggle for styles
