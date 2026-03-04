"""Canned transcript data for the Lucea Health triage demo.

Simulates a pediatric triage call: 7-year-old Emma Martinez with known
asthma presenting with nighttime coughing.  Three AI agents assist the
nurse in real-time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranscriptEvent:
    """A single event in the triage call timeline."""

    timestamp: str
    panel: str  # "left" or "right"
    speaker: str  # nurse | guardian | nurse-assistant | patient-agent | guardian-agent
    content: str
    event_type: str  # message | context | protocol | alert
    delay_ms: int = 800


# -- Canned transcript --------------------------------------------------------

TRIAGE_TRANSCRIPT: list[TranscriptEvent] = [
    # -- PRE-CALL: Agent hydration (right panel) ------------------------------
    TranscriptEvent(
        timestamp="0:00",
        panel="right",
        speaker="patient-agent",
        content=(
            "<b>Patient Loaded — Emma Martinez</b><br>"
            "Age 7 · 23 kg (50th %ile) · MRN BFP-2024-08812<br>"
            "<b>Active:</b> Asthma, moderate persistent (J45.40)<br>"
            "<b>Meds:</b> Fluticasone 44 mcg 2 puffs BID · Albuterol PRN<br>"
            "<b>Allergies:</b> NKDA<br>"
            "<b>⚠ Recent ER visit 02/12/2026</b> — acute exacerbation, "
            "prednisolone taper completed"
        ),
        event_type="context",
        delay_ms=600,
    ),
    TranscriptEvent(
        timestamp="0:01",
        panel="right",
        speaker="guardian-agent",
        content=(
            "<b>Guardian — Rosa Martinez (Mother)</b><br>"
            "Preferred language: English · Communication: direct, no jargon<br>"
            "4 calls in 6 months · <b>Pattern:</b> escalates if concerns minimized<br>"
            "<b>Previous frustration:</b> ER wait on 02/12 — validate concerns early<br>"
            "Best callback: mornings 8-10 AM or evenings after 7:30 PM"
        ),
        event_type="context",
        delay_ms=800,
    ),
    # -- CALL START -----------------------------------------------------------
    TranscriptEvent(
        timestamp="0:03",
        panel="left",
        speaker="nurse-assistant",
        content=(
            "• <b>Incoming call:</b> Rosa Martinez — re: Emma Martinez (7yr)<br>"
            "• 4th call in 6 months — ER visit 3 weeks ago (acute exacerbation)<br>"
            "• Prednisolone taper completed 02/17 — monitor for rebound symptoms<br>"
            "• Guardian responds well to validation before clinical questions"
        ),
        event_type="context",
        delay_ms=1200,
    ),
    TranscriptEvent(
        timestamp="0:05",
        panel="left",
        speaker="nurse",
        content=(
            "Good evening, Mrs. Martinez, this is Sarah with Bright Futures "
            "Pediatrics. I can see Emma's chart here — how can I help tonight?"
        ),
        event_type="message",
        delay_ms=1000,
    ),
    TranscriptEvent(
        timestamp="0:12",
        panel="right",
        speaker="guardian",
        content=(
            "Hi Sarah. Emma's been coughing a lot the last two nights — it's "
            "worse when she's lying down. She's waking up at least three or "
            "four times. I'm worried because we were just in the ER last month "
            "and I don't want this to turn into another episode like that."
        ),
        event_type="message",
        delay_ms=1400,
    ),
    # -- NURSE ASSISTANT: Protocol surfacing -----------------------------------
    TranscriptEvent(
        timestamp="0:18",
        panel="left",
        speaker="nurse-assistant",
        content=(
            "<b>Schmitt-Thompson Protocol 42B — Cough, Age 1-11</b><br>"
            "Key differentiators to assess:<br>"
            "• Rescue inhaler usage in past 48 hrs<br>"
            "• Fever presence (Y/N) — separates URI-driven vs. asthma-driven<br>"
            "• Work of breathing: retractions, nasal flaring, tripod positioning<br>"
            "• SpO2 if home oximeter available<br><br>"
            "<b>Context:</b> Post-ER exacerbation (02/12) — nocturnal cough within "
            "30 days is a relapse signal per protocol"
        ),
        event_type="protocol",
        delay_ms=1000,
    ),
    # -- SYMPTOM DISCUSSION ---------------------------------------------------
    TranscriptEvent(
        timestamp="0:25",
        panel="left",
        speaker="nurse",
        content=(
            "I completely understand your concern — especially after last "
            "month. Let's figure out what's going on together. Can you tell "
            "me, has Emma needed her rescue inhaler in the last two days?"
        ),
        event_type="message",
        delay_ms=1200,
    ),
    TranscriptEvent(
        timestamp="0:35",
        panel="right",
        speaker="guardian",
        content=(
            "Yes, I gave her two puffs last night around 2 AM and again "
            "tonight around midnight. It helped for maybe an hour each time "
            "but the cough came back."
        ),
        event_type="message",
        delay_ms=1200,
    ),
    TranscriptEvent(
        timestamp="0:40",
        panel="left",
        speaker="nurse-assistant",
        content=(
            "⚠ <b>Rescue inhaler usage elevated</b><br>"
            "• Baseline: 1-2x/week → Current: 2x in 24 hrs (nightly)<br>"
            "• Relief duration &lt;1 hr suggests incomplete bronchodilation<br>"
            "• Per asthma action plan: this pattern = <b>Yellow Zone</b>"
        ),
        event_type="alert",
        delay_ms=800,
    ),
    TranscriptEvent(
        timestamp="0:45",
        panel="left",
        speaker="nurse",
        content=(
            "Okay, that's really helpful. Has she had any fever at all — "
            "even low-grade?"
        ),
        event_type="message",
        delay_ms=1000,
    ),
    TranscriptEvent(
        timestamp="0:50",
        panel="right",
        speaker="guardian",
        content=(
            "I checked tonight before I called — 99.1. So just barely. "
            "She doesn't seem sick otherwise, she was fine at school today."
        ),
        event_type="message",
        delay_ms=1000,
    ),
    TranscriptEvent(
        timestamp="0:55",
        panel="left",
        speaker="nurse-assistant",
        content=(
            "• Temp 99.1°F — low-grade, not significant for infection threshold<br>"
            "• Active at school today — no daytime functional impairment<br>"
            "• Pattern: <b>isolated nocturnal cough + increased rescue use</b><br>"
            "• Consistent with asthma flare vs. URI-triggered exacerbation<br>"
            "• Lucas (sibling, age 3) in daycare — URI exposure source possible"
        ),
        event_type="context",
        delay_ms=800,
    ),
    # -- ASSESSMENT -----------------------------------------------------------
    TranscriptEvent(
        timestamp="1:00",
        panel="left",
        speaker="nurse",
        content=(
            "One more question — when she's coughing, are you noticing any "
            "fast breathing, her ribs pulling in, or her sitting up to "
            "breathe more easily?"
        ),
        event_type="message",
        delay_ms=1200,
    ),
    TranscriptEvent(
        timestamp="1:08",
        panel="right",
        speaker="guardian",
        content=(
            "No, nothing like that. She does prop herself up on the pillow "
            "and the coughing stops for a bit, but no pulling or fast "
            "breathing. She can talk normally between coughs."
        ),
        event_type="message",
        delay_ms=1200,
    ),
    TranscriptEvent(
        timestamp="1:12",
        panel="left",
        speaker="nurse-assistant",
        content=(
            "<b>Assessment Summary — Protocol 42B</b><br>"
            "✓ No respiratory distress signs (no retractions, nasal flaring)<br>"
            "✓ Can speak in full sentences between coughs<br>"
            "✓ Positional relief (upright) — classic nocturnal asthma pattern<br>"
            "⚠ Rescue inhaler &lt;1 hr relief = Yellow Zone on action plan<br>"
            "⚠ 3 weeks post-ER exacerbation — relapse window<br><br>"
            "<b>Protocol-aligned dispositions:</b><br>"
            "• Home care with step-up guidance (Yellow Zone protocol)<br>"
            "• Office visit within 24 hrs if not improving<br>"
            "• ER if work of breathing changes or rescue inhaler stops helping"
        ),
        event_type="protocol",
        delay_ms=1000,
    ),
    # -- DISPOSITION ----------------------------------------------------------
    TranscriptEvent(
        timestamp="1:20",
        panel="left",
        speaker="nurse",
        content=(
            "Okay Mrs. Martinez, here's what I'm seeing. Emma's symptoms "
            "line up with her asthma flaring — the nighttime cough and needing "
            "her rescue inhaler more often. The good news is she's not showing "
            "signs of distress right now. Here's what I'd like you to do tonight:"
        ),
        event_type="message",
        delay_ms=1400,
    ),
    TranscriptEvent(
        timestamp="1:30",
        panel="left",
        speaker="nurse",
        content=(
            "First, give her two puffs of the albuterol with the spacer now "
            "and elevate her head with an extra pillow. Second, if she wakes "
            "up coughing again, you can repeat the albuterol every four hours. "
            "Third — and this is important — if the inhaler stops helping or "
            "you notice her breathing fast or working hard to breathe, go to "
            "the ER right away. I'm going to have Dr. Okonkwo's office call "
            "you first thing in the morning to get her seen tomorrow."
        ),
        event_type="message",
        delay_ms=1600,
    ),
    TranscriptEvent(
        timestamp="1:38",
        panel="right",
        speaker="guardian",
        content=(
            "That makes sense. So albuterol now, extra pillow, repeat every "
            "four hours if she wakes up, and ER if the inhaler stops working. "
            "And they'll call me in the morning?"
        ),
        event_type="message",
        delay_ms=1000,
    ),
    TranscriptEvent(
        timestamp="1:42",
        panel="left",
        speaker="nurse-assistant",
        content=(
            "✓ <b>Teach-back confirmed</b> — guardian repeated all key instructions<br>"
            "• Albuterol dosing: 2 puffs q4h PRN (weight-appropriate for 23 kg)<br>"
            "• Callback note flagged for Dr. Okonkwo's morning schedule"
        ),
        event_type="context",
        delay_ms=800,
    ),
    TranscriptEvent(
        timestamp="1:45",
        panel="left",
        speaker="nurse",
        content=(
            "Exactly right. And Mrs. Martinez — you did the right thing "
            "calling tonight. With Emma's history, it's always better to "
            "check in early. Don't hesitate to call back if anything changes."
        ),
        event_type="message",
        delay_ms=1200,
    ),
    TranscriptEvent(
        timestamp="1:50",
        panel="right",
        speaker="guardian",
        content="Thank you, Sarah. I appreciate you taking the time. Good night.",
        event_type="message",
        delay_ms=800,
    ),
    # -- DOCUMENTATION --------------------------------------------------------
    TranscriptEvent(
        timestamp="1:55",
        panel="left",
        speaker="nurse-assistant",
        content=(
            "<b>Draft Clinical Note — SBAR Format</b><br><br>"
            "<b>Situation:</b> After-hours triage call from mother (Rosa Martinez) "
            "for Emma Martinez, 7yo F with moderate persistent asthma. "
            "Presenting with 2-night history of nocturnal cough, increased "
            "rescue inhaler use.<br><br>"
            "<b>Background:</b> ER visit 02/12/2026 for acute exacerbation "
            "(prednisolone taper completed 02/17). Controller medication "
            "(fluticasone 44 mcg BID) reportedly adherent. Baseline rescue "
            "use 1-2x/week, now 2x/24hrs with &lt;1 hr relief.<br><br>"
            "<b>Assessment:</b> Nocturnal cough with increased rescue inhaler use, "
            "consistent with Yellow Zone per asthma action plan. No respiratory "
            "distress. Low-grade temp 99.1°F. Functional at school. "
            "Pattern suggests early asthma flare, possible URI trigger "
            "(sibling in daycare).<br><br>"
            "<b>Recommendation:</b> Home management with rescue inhaler q4h PRN, "
            "positional comfort. ER precautions given. Follow-up with "
            "Dr. Okonkwo within 24 hrs — callback flagged for morning schedule. "
            "Schmitt-Thompson Protocol 42B applied."
        ),
        event_type="context",
        delay_ms=1200,
    ),
]
