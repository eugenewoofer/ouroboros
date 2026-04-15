"""Regression checks for chat live-card and grouped logs UI."""

import os
import pathlib
import re

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_chat_progress_updates_route_into_live_card():
    source = _read("web/modules/chat.js")

    assert "const liveCardRecords = new Map();" in source
    assert "const taskUiStates = new Map();" in source
    assert "summarizeChatLiveEvent" in source
    assert "Show details" in source
    assert "if (msg.is_progress) {" in source
    assert "updateLiveCardFromProgressMessage(msg);" in source
    assert "appendTaskSummaryToLiveCard(msg);" in source
    assert "if (explicitTaskId) finishLiveCard(explicitTaskId);" in source
    assert "ws.on('log', (msg) => {" in source
    assert "updateLiveCardFromLogEvent(msg.data);" in source
    assert "if (msg.is_progress) {" in source
    assert "hideTypingIndicatorOnly();" in source
    assert "function hasActiveLiveCard()" in source
    assert "state.activePage !== 'chat'" in source
    assert "function isNearBottom(threshold = 96)" in source
    assert "if (node.parentNode === messagesDiv) {" in source
    assert "if (shouldStick) messagesDiv.scrollTop = messagesDiv.scrollHeight;" in source
    assert "function markTaskToolCall(taskId, count = 1, minimumOnly = false)" in source
    assert "taskState.forceCard || taskState.toolCalls > 1 || shouldAlwaysShowTaskCard(taskState.taskId)" in source
    assert "function forceTaskCard(taskId)" in source
    assert "function markAssistantReply(taskId = '')" in source
    assert "function isTerminalTaskPhase(phase = '') {" in source
    assert "taskState.completed = true;" in source
    assert "scheduleTaskUiCleanup(taskState, 30000);" in source
    assert "if (taskState.completed && !isTerminalTaskPhase(summary.phase || '')) {" in source
    assert "if (record.finished && !isTerminalTaskPhase(nextPhase)) {" in source
    assert "const taskId = msg.task_id || '';" in source
    assert "const taskState = getTaskUiState(taskId, true);" in source
    assert "const wasFinished = record.finished;" in source
    assert "const justFinished = record.finished && !wasFinished;" in source
    assert "if (justFinished) {" in source
    assert "if (!wasFinished) {" in source
    assert "function setLiveCardExpanded(record, expanded) {" in source
    assert "record.root.dataset.expanded = expanded ? '1' : '0';" in source
    assert "function syncLiveCardLayout(record) {" in source
    assert "record.root.style.minHeight = `${Math.max(summaryHeight + timelineHeight, 0)}px`;" in source


def test_live_card_recovery_keeps_step_failures_non_terminal():
    chat_source = _read("web/modules/chat.js")
    log_source = _read("web/modules/log_events.js")

    assert "return phase === 'done';" in chat_source
    assert "if (phase === 'warn') return 'Notice';" in chat_source
    assert "record.finished = isTerminalTaskPhase(nextPhase);" in chat_source
    assert "const activePhase = ['error', 'timeout'].includes(phase) ? phase : 'done';" in chat_source
    assert "function extractCommandText(args) {" in log_source
    assert "evt.status === 'non_zero_exit'" in log_source
    assert "phase: 'warn'" in log_source
    assert "A command returned" in log_source
    assert "commandText.full || errorResult.full" in log_source


def test_logs_use_shared_log_event_helpers_and_group_task_cards():
    logs_source = _read("web/modules/logs.js")
    shared_source = _read("web/modules/log_events.js")

    assert "from './log_events.js'" in logs_source
    assert "isGroupedTaskEvent(evt)" in logs_source
    assert "createTaskGroupCard" in logs_source
    assert "renderTaskTimeline" in logs_source
    assert "export function summarizeLogEvent" in shared_source
    assert "export function summarizeChatLiveEvent" in shared_source
    assert "export function isGroupedTaskEvent" in shared_source
    assert "export function getLogTaskGroupId" in shared_source


def test_styles_cover_chat_header_controls_and_grouped_cards():
    css = _read("web/style.css")

    assert "--accent-light:" in css
    assert ".chat-header-actions {" in css
    assert ".chat-header-btn {" in css
    assert ".chat-live-card {" in css
    assert '.chat-live-card[data-finished="1"] {' in css
    assert ".chat-live-timeline {" in css
    assert ".chat-live-toggle {" in css
    assert ".chat-live-summary-button {" in css
    assert '.chat-live-card[data-expanded="1"] .chat-live-chevron {' in css
    assert '.chat-live-card[data-expanded="1"] .chat-live-timeline {' in css
    assert ".log-task-card {" in css
    assert ".log-task-timeline {" in css
    assert re.search(r"\.chat-live-title\s*\{[^}]*font-weight:\s*400;", css, re.S)
    assert re.search(r"\.chat-live-line-title\s*\{[^}]*font-weight:\s*400;", css, re.S)
    assert re.search(r"\.chat-live-line-body\s*\{[^}]*font-size:\s*\d+px;", css, re.S)


def test_chat_only_polls_state_when_active():
    chat_source = _read("web/modules/chat.js")

    assert "state.activePage !== 'chat'" in chat_source


def test_live_card_has_inline_typing_dots_and_pulsing_phase_badge():
    css = _read("web/style.css")
    chat_source = _read("web/modules/chat.js")

    # Inline typing dots in live card summary
    assert ".chat-live-typing {" in css
    assert ".chat-live-typing span {" in css
    assert 'animation: typing-bounce' in css
    assert '.chat-live-card[data-finished="1"] .chat-live-typing {' in css

    # Active phase badge should pulse
    assert "animation: thinking-pulse" in css
    assert '.chat-live-card:not([data-finished="1"]) .chat-live-phase.working' in css

    # JS: typing visibility helpers exist
    assert "function setLiveCardTypingVisible(record, visible) {" in chat_source
    assert "inlineTypingEl: root.querySelector('[data-live-typing]')" in chat_source
    assert "setLiveCardTypingVisible(record, false);" in chat_source
    assert "setLiveCardTypingVisible(record, true);" in chat_source

    # HTML template has the typing element
    assert "data-live-typing" in chat_source


def test_live_card_timeline_body_renders_markdown():
    """Live card timeline body must use renderMarkdown, not escapeHtml."""
    source = _read("web/modules/chat.js")
    # The body div inside the timeline must use renderMarkdown so that
    # markdown formatting (bold, lists, etc.) is rendered correctly.
    assert "renderMarkdown(displayBody)" in source
    # escapeHtml must NOT be used for displayBody in the timeline
    assert "escapeHtml(displayBody)" not in source


def test_live_card_timeline_headline_renders_markdown_for_progress():
    """working/thinking phase lines must render their headline with renderMarkdown."""
    source = _read("web/modules/chat.js")
    # isProgressLine must check the actual phase values emitted by summarizeChatLiveEvent
    assert "isProgressLine" in source
    assert "isProgressLine ? renderMarkdown(displayHeadline)" in source
    # Must use 'working' and 'thinking' — NOT the dead 'progress'/'thought' names
    assert "item.phase === 'working' || item.phase === 'thinking'" in source
    assert "item.phase === 'progress' || item.phase === 'thought'" not in source


def test_chat_history_replays_task_summaries_into_live_cards():
    history_source = _read("ouroboros/server_history_api.py")
    chat_source = _read("web/modules/chat.js")

    assert '"task_id": str(entry.get("task_id", ""))' in history_source
    assert "const taskId = msg.task_id || '';" in chat_source
    assert "appendTaskSummaryToLiveCard(msg);" in chat_source
    assert "taskId," in chat_source
    assert "if (role !== 'user' && !opts.isProgress && opts.taskId) {" in chat_source


def test_chat_history_conditionally_forces_live_card_for_task_summaries():
    """Historical task_summary messages must force the live card visible
    only for non-trivial tasks (tool_calls > 0 or rounds > 1).

    After a restart taskState.toolCalls is 0, so revealBufferedCardIfNeeded
    would silently skip the card unless forceCard is set.  But trivial tasks
    (simple replies) should not show a card at all.
    """
    source = _read("web/modules/chat.js")
    # forceCard is conditional — only set when the task was non-trivial
    assert "taskState.forceCard = true;" in source
    assert "(msg.tool_calls || 0) > 0" in source
    assert "(msg.rounds || 0) > 1" in source
    # The condition + force must happen inside the task_summary branch of syncHistory
    idx_force = source.index("taskState.forceCard = true;")
    idx_append = source.index("appendTaskSummaryToLiveCard(msg);")
    assert idx_force < idx_append, "forceCard must be set before appendTaskSummaryToLiveCard"


def test_progress_messages_force_live_card_visible():
    """Progress messages (e.g. '🔍 Searching...') must force the live card open.

    Without forceCard, a single-tool-call task (like web_search) would buffer
    progress into an invisible card (toolCalls <= 1) and the user sees nothing
    until the final result arrives. forceCard must be set inside
    updateLiveCardFromProgressMessage before queueTaskLiveUpdate is called.
    """
    source = _read("web/modules/chat.js")
    # Find the updateLiveCardFromProgressMessage function
    func_start = source.index("function updateLiveCardFromProgressMessage(")
    # Find the next function definition to bound the search
    func_body_end = source.index("\n    function ", func_start + 1)
    func_body = source[func_start:func_body_end]
    # forceCard must be set to true
    assert "taskState.forceCard = true" in func_body, \
        "updateLiveCardFromProgressMessage must set forceCard = true"
    # forceCard must be set BEFORE queueTaskLiveUpdate
    idx_force = func_body.index("forceCard = true")
    idx_queue = func_body.index("queueTaskLiveUpdate(")
    assert idx_force < idx_queue, \
        "forceCard must be set before queueTaskLiveUpdate in updateLiveCardFromProgressMessage"


def test_task_summary_live_card_uses_last_headline_not_finished_task():
    """appendTaskSummaryToLiveCard must NOT use 'Finished task' as headline.

    Instead it should use lastHumanHeadline from the live card record
    and not add a visible timeline entry (the summary text duplicates
    the assistant reply bubble).
    """
    source = _read("web/modules/chat.js")
    func_start = source.index("function appendTaskSummaryToLiveCard(")
    func_body_end = source.index("\n    function ", func_start + 1)
    func_body = source[func_start:func_body_end]
    # Must NOT contain "Finished task" headline
    assert "Finished task" not in func_body, \
        "appendTaskSummaryToLiveCard should not use 'Finished task' as headline"
    # Must use lastHumanHeadline
    assert "lastHumanHeadline" in func_body, \
        "appendTaskSummaryToLiveCard should use record.lastHumanHeadline"
    # Must set visible: false to avoid adding a timeline entry
    assert "visible: false" in func_body, \
        "appendTaskSummaryToLiveCard should set visible: false"


def test_chat_input_has_glassmorphism():
    """#chat-input textarea should have frosted-glass styling (blur + semi-transparent bg + white border)."""
    css = _read("web/style.css")
    # Extract the #chat-input block (between selector and :focus)
    start = css.index("#chat-input {")
    end = css.index("#chat-input:focus")
    chat_input_block = css[start:end]
    # Must have high-quality backdrop blur on the textarea itself
    assert "backdrop-filter: blur(16px)" in chat_input_block
    # Background should be semi-transparent (frosted glass, opacity 0.62)
    assert "rgba(26, 21, 32, 0.62)" in chat_input_block
    # Border should be a white tint (design system for frosted glass surfaces)
    assert "rgba(255, 255, 255, 0.09)" in chat_input_block
    # Must not use opaque background
    assert "var(--bg-secondary)" not in chat_input_block


def test_log_phases_use_crimson_not_blue():
    """Active log phases (start/progress/working/thinking) should use crimson, not blue."""
    css = _read("web/style.css")
    # Find the active-phase block
    assert "log-phase.working" in css
    assert "log-phase.thinking" in css
    # The active-phase color must be crimson, not --blue
    active_block_start = css.index(".log-entry .log-phase.start,")
    active_block_end = css.index("}", active_block_start)
    active_block = css[active_block_start:active_block_end]
    assert "var(--blue)" not in active_block
    assert "rgba(248, 130, 140" in active_block


def test_about_uses_css_classes_not_inline():
    """about.js must use CSS classes, not inline style= attributes."""
    src = _read("web/modules/about.js")
    assert 'class="about-body"' in src
    assert 'class="about-logo"' in src
    assert 'class="about-title"' in src
    assert 'class="about-credits"' in src
    assert 'class="about-footer"' in src
    # No inline style= should remain
    assert 'style="' not in src


def test_costs_uses_css_classes_not_inline():
    """costs.js must use CSS classes for layout; bar width via DOM .style.width is acceptable."""
    src = _read("web/modules/costs.js")
    # Static HTML uses class= attributes
    assert 'class="costs-stats-grid"' in src
    assert 'class="costs-tables-grid"' in src
    assert 'class="costs-table-label"' in src
    # DOM-built cells use .className assignments (not class= in innerHTML)
    assert "className = 'cost-cell-name'" in src
    assert "className = 'cost-bar'" in src
    # renderBreakdownTable must use DOM creation (no innerHTML for user data)
    assert "document.createElement('tr')" in src
    assert "textContent = name" in src
    # Only the dynamic bar width remains as .style.width — that's the sole acceptable exception
    assert "bar.style.width" in src


def test_costs_about_css_classes_defined():
    """All CSS classes used by about.js and costs.js must be defined in style.css."""
    css = _read("web/style.css")
    for cls in [".about-body", ".about-logo", ".about-title", ".about-footer",
                ".costs-stats-grid", ".costs-tables-grid", ".costs-table-label",
                ".cost-cell-name", ".cost-bar-cell", ".cost-bar", ".cost-empty-cell"]:
        assert cls in css, f"Missing CSS class: {cls}"


def test_trivial_task_summary_skip_in_backend():
    """_run_task_summary must skip the LLM call for trivial tasks (0 tool calls, ≤1 round)."""
    source = (REPO / "ouroboros" / "agent_task_pipeline.py").read_text(encoding="utf-8")
    # Early return for trivial tasks
    assert "n_tool_calls == 0 and rounds <= 1" in source
    # Metadata must be written to chat.jsonl even for trivial tasks
    assert '"tool_calls": n_tool_calls' in source
    assert '"rounds": rounds' in source


def test_history_api_passes_task_summary_metadata():
    """server_history_api must pass tool_calls and rounds fields for task_summary entries."""
    source = (REPO / "ouroboros" / "server_history_api.py").read_text(encoding="utf-8")
    assert 'entry.get("type") == "task_summary"' in source
    assert 'rec["tool_calls"]' in source
    assert 'rec["rounds"]' in source


# ---------------------------------------------------------------------------
# Evolution Versions sub-tab: no inline styles, CSS-class-based structure
# ---------------------------------------------------------------------------

def test_evolution_versions_subtab_uses_css_classes_not_inline_styles():
    """Evolution Versions sub-tab must use CSS classes, not inline style="" attributes."""
    source = _read("web/modules/evolution.js")

    # Container and layout classes must be present
    assert 'class="evo-versions-content"' in source
    assert 'class="evo-versions-header"' in source
    assert 'class="evo-versions-branch"' in source
    assert 'class="evo-versions-cols"' in source
    assert 'class="evo-versions-col"' in source

    # Row helper must use CSS classes, not inline styles
    assert 'evo-versions-row' in source
    assert 'evo-versions-row-label' in source
    assert 'evo-versions-row-msg' in source
    assert 'btn-xs' in source

    # Error/empty states must use CSS class, not inline style
    assert 'evo-empty-error' in source

    import re
    # renderRow innerHTML must not contain inline style= attributes
    # (row.style.xxx JS property assignments are irrelevant — only template literal innerHTML is checked)
    render_row_match = re.search(
        r'function renderRow\(.*?\{(.*?)\n    \}', source, re.DOTALL
    )
    assert render_row_match, "renderRow function not found in evolution.js"
    render_row_body = render_row_match.group(1)
    inner_html_parts = re.findall(r'innerHTML\s*=\s*`(.+?)`', render_row_body, re.DOTALL)
    assert inner_html_parts, "renderRow should set innerHTML via template literal"
    for part in inner_html_parts:
        assert 'style=' not in part, f"renderRow innerHTML still contains inline style=: {part[:120]}"

    # Versions sub-tab container HTML: confirm class-based markers are present and no inline styles
    # Extract between the two known comment anchors that exist in the file
    versions_block_match = re.search(
        r'<!-- Versions sub-tab -->(.*?)<!-- Chart sub-tab|<!-- Versions sub-tab -->(.*?)$',
        source, re.DOTALL
    )
    if versions_block_match:
        versions_html = versions_block_match.group(1) or versions_block_match.group(2) or ''
        # Verify class markers are present
        assert 'evo-versions-content' in versions_html
        assert 'evo-versions-header' in versions_html
        # Verify no inline style= in the static template HTML
        # (JS .style property assignments are not in the template string)
        template_literal_match = re.search(r'page\.innerHTML\s*=\s*`(.*?)`\s*;', source, re.DOTALL)
        if template_literal_match:
            template = template_literal_match.group(1)
            # Find the versions section in the template and assert no style= there
            versions_in_template = template[template.find('<!-- Versions sub-tab -->'):]
            assert 'style=' not in versions_in_template[:2000], \
                "Versions sub-tab template still contains inline style= attributes"


def test_evolution_versions_css_classes_defined():
    """CSS classes introduced for Versions sub-tab must be defined in style.css."""
    css = _read("web/style.css")
    for cls in [
        ".evo-versions-content",
        ".evo-versions-header",
        ".evo-versions-branch",
        ".evo-versions-cols",
        ".evo-versions-col",
        ".evo-versions-list",
        ".evo-versions-row",
        ".evo-versions-row-label",
        ".evo-versions-row-msg",
        ".evo-empty-error",
        ".btn-xs",
    ]:
        assert cls in css, f"Missing CSS class in style.css: {cls}"


def test_evolution_load_versions_error_handling():
    """loadVersions() must guard resp.ok and clear all three UI surfaces on error."""
    source = _read("web/modules/evolution.js")

    # Must check resp.ok before parsing JSON (guards non-2xx responses)
    assert "if (!resp.ok) throw new Error" in source, \
        "loadVersions must throw on non-OK HTTP response"

    # Extract the loadVersions function body (up to rollback)
    load_versions_start = source.find("async function loadVersions()")
    rollback_start = source.find("async function rollback(", load_versions_start)
    assert load_versions_start != -1 and rollback_start != -1
    fn_body = source[load_versions_start:rollback_start]

    # The catch branch must clear all three surfaces
    catch_start = fn_body.rfind("} catch (e) {")
    assert catch_start != -1, "loadVersions catch block not found"
    catch_body = fn_body[catch_start:]
    assert 'commitsDiv.innerHTML' in catch_body, "catch must clear commitsDiv"
    assert 'tagsDiv.innerHTML' in catch_body, "catch must clear tagsDiv"
    assert 'currentDiv.textContent' in catch_body, "catch must reset currentDiv"
    assert 'versionsLoaded = false' in catch_body, "catch must reset versionsLoaded"


def test_evolution_runtime_card_uses_crimson_border():
    """evo-runtime-card and evo-chart-wrap must use crimson accent border, not neutral white."""
    css = _read("web/style.css")
    # The rule should contain a crimson rgba border, not the old neutral white
    import re
    rule_match = re.search(
        r'\.evo-runtime-card,\s*\.evo-chart-wrap\s*\{(.+?)\}', css, re.DOTALL
    )
    assert rule_match, ".evo-runtime-card rule not found in style.css"
    rule_body = rule_match.group(1)
    # Must have a crimson border (201, 53, 69)
    assert "201, 53, 69" in rule_body, "evo-runtime-card should use crimson accent border"
    # Must NOT use the old neutral white border
    assert "255, 255, 255, 0.08" not in rule_body, "evo-runtime-card should not use neutral white border"


def test_live_card_timeline_no_hardcoded_item_cap():
    """Live card timeline must not drop items — no 20-item shift() cap."""
    source = _read("web/modules/chat.js")

    # The old hard cap must be gone
    assert "record.items.length > 20" not in source, "20-item cap must be removed from record.items"
    assert "bufferedLiveUpdates.length > 20" not in source, "20-item cap must be removed from bufferedLiveUpdates"

    # Incremental rendering helpers must be present
    assert "function buildTimelineItemHtml(item, record)" in source
    assert "function appendTimelineItem(item, record)" in source
    assert "function patchLastTimelineItem(item, record)" in source

    # timelineUpdate flag drives incremental vs full-rebuild path
    assert "timelineUpdate = 'append'" in source
    assert "timelineUpdate = 'patch-last'" in source
    assert "appendTimelineItem(lastItem, record)" in source
    assert "patchLastTimelineItem(lastItem, record)" in source

    # TIMELINE_MAX_HEIGHT constant must be defined
    assert "const TIMELINE_MAX_HEIGHT = 420;" in source
    # syncLiveCardLayout must clamp to it
    assert "Math.min(" in source and "TIMELINE_MAX_HEIGHT" in source

    # Memory release must only free completed cards (guards rec.finished)
    assert "rec.root?.remove()" in source
    assert "rec && rec.finished" in source
    # Retired set prevents syncHistory from recreating cleaned tasks
    assert "const retiredTaskIds = new Set();" in source
    assert "retiredTaskIds.add(taskState.taskId);" in source
    assert "if (retiredTaskIds.has(taskId)) continue;" in source
    # Reusable ids (bg-consciousness, active) reset on new cycle, not retired
    assert "const REUSABLE_TASK_IDS = new Set(" in source
    assert "REUSABLE_TASK_IDS.has(resolvedTaskId)" in source
    assert "taskState.completed = false;" in source


def test_chat_input_overlay_buttons_css():
    """Regression: paperclip and Send buttons must be absolute overlays inside the textarea."""
    css = _read("web/style.css")

    # .chat-attach-btn: absolute, inside input wrap on the left
    attach_match = re.search(
        r'\.chat-attach-btn\s*\{(.+?)\}',
        css,
        re.DOTALL,
    )
    assert attach_match, ".chat-attach-btn CSS rule not found"
    attach_body = attach_match.group(1)
    assert "position: absolute" in attach_body, ".chat-attach-btn must be position: absolute"
    assert "left:" in attach_body, ".chat-attach-btn must have a left: offset"

    # .chat-send-inline: absolute, inside input wrap on the right
    send_match = re.search(
        r'\.chat-send-inline\s*\{(.+?)\}',
        css,
        re.DOTALL,
    )
    assert send_match, ".chat-send-inline CSS rule not found"
    send_body = send_match.group(1)
    assert "position: absolute" in send_body, ".chat-send-inline must be position: absolute"
    assert "right:" in send_body, ".chat-send-inline must have a right: offset"

    # #chat-input: must have both left and right padding to avoid text overlap
    input_match = re.search(
        r'#chat-input\s*\{(.+?)\}',
        css,
        re.DOTALL,
    )
    assert input_match, "#chat-input CSS rule not found"
    input_body = input_match.group(1)
    assert "padding" in input_body, "#chat-input must have padding defined"
    # padding: 10px 52px 10px 42px — right 52px, left 42px
    # We check that the padding value is not the symmetric default
    assert "42px" in input_body, "#chat-input must have 42px left padding for attach overlay"
    assert "52px" in input_body, "#chat-input must have 52px right padding for send overlay"


def test_sync_history_two_pass_progress_before_finalize():
    """syncHistory must use two-pass processing: progress/summary first, then regular messages.

    This guarantees that progress bubbles are not discarded when taskState.completed=true
    is set by a prior assistant reply during the same sync iteration.
    """
    source = _read("web/modules/chat.js")

    # Two-pass comment markers must be present
    assert "Two-pass processing" in source, \
        "syncHistory must use two-pass processing"
    assert "Pass 1: progress messages and task summaries" in source, \
        "syncHistory must have an explicit Pass 1 comment"
    assert "Pass 2: regular messages" in source, \
        "syncHistory must have an explicit Pass 2 comment"

    # Pass 2 must skip progress and task_summary (already handled)
    assert "if (msg.is_progress) continue;" in source, \
        "Pass 2 must skip is_progress messages"
    assert "if (msg.system_type === 'task_summary') continue;" in source, \
        "Pass 2 must skip task_summary messages"

    # Pass 1 must NOT check taskState.completed before calling updateLiveCardFromProgressMessage
    # (the old guard that caused bubbles to be lost)
    func_start = source.index("async function syncHistory(")
    func_end = source.index("\n    async function ", func_start + 1) if "\n    async function " in source[func_start + 1:] else len(source)
    sync_body = source[func_start:func_end]

    # The two-pass structure means the old pattern is gone
    # Verify there is no guard between Pass-1 start and updateLiveCardFromProgressMessage
    pass1_start = sync_body.index("Pass 1: progress messages")
    pass2_start = sync_body.index("Pass 2: regular messages")
    pass1_body = sync_body[pass1_start:pass2_start]

    assert "if (taskState.completed) continue;" not in pass1_body, \
        "Pass 1 must not skip progress messages due to taskState.completed"


def test_sync_history_shows_typing_for_ongoing_tasks():
    """syncHistory must call showTyping() after first load if a live card is still active."""
    source = _read("web/modules/chat.js")

    # The post-load typing check must be present
    assert "After first load" in source, \
        "syncHistory must have a post-load typing indicator check"
    assert "const hasOngoingTask = Array.from(liveCardRecords.values()).some(" in source, \
        "Must check for active live cards after history load"
    assert "if (hasOngoingTask) showTyping();" in source, \
        "Must call showTyping() when an ongoing task is detected"
    # Guard: only on first load (!historyLoaded)
    assert "if (!historyLoaded) {" in source, \
        "Typing restore must only happen on first page load"


def test_update_live_card_no_forcecard_for_completed():
    """updateLiveCardFromProgressMessage must not force-open cards for completed tasks."""
    source = _read("web/modules/chat.js")

    func_start = source.index("function updateLiveCardFromProgressMessage(")
    func_end = source.index("\n    function ", func_start + 1)
    func_body = source[func_start:func_end]

    # Must have the guard !taskState.completed before setting forceCard
    assert "if (taskState && !taskState.completed) taskState.forceCard = true;" in func_body, \
        "updateLiveCardFromProgressMessage must not force-open cards for already-completed tasks"


def test_live_card_timeline_css_scrollable():
    """Expanded chat live timeline must be scrollable with a max-height."""
    import re
    css = _read("web/style.css")

    # Find the expanded timeline rule
    rule_match = re.search(
        r'\.chat-live-card\[data-expanded="1"\]\s*\.chat-live-timeline\s*\{(.+?)\}',
        css,
        re.DOTALL,
    )
    assert rule_match, ".chat-live-card[data-expanded='1'] .chat-live-timeline rule not found"
    rule_body = rule_match.group(1)

    assert "max-height" in rule_body, "timeline must have max-height when expanded"
    assert "420px" in rule_body, "timeline max-height must be 420px"
    assert "overflow-y" in rule_body, "timeline must have overflow-y for scrolling"
    assert "auto" in rule_body, "overflow-y must be auto"
