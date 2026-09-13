"""Declarative specification of the context-rot benchmark (plan Phase 7).

Everything that is *content* lives here: the fact vocabulary, the sentence
templates, the seven categories, and the length ladder. Everything that is
*structure* — where a fact sits in a conversation, how long the filler runs, how
the ledger is assembled — lives in :mod:`benchmarks.context_rot.generation`.

Keeping the two apart is what makes the generator reviewable: a reviewer can
check whether the benchmark asks the right questions without reading the packing
algorithm, and a new category is a template plus one plan function rather than a
rewrite.

Design constraints the templates satisfy (and that
``tests/evaluation/test_context_rot_dataset.py`` enforces):

* **A fact must be storable.** Every planted fact is a declarative sentence the
  application's memory policy classifies as memory (a verb from the policy's
  statement set, no question mark, no filler marker). A fact the policy ignores
  would make BrainOS mode fail for a reason that is not under test.
* **Filler must not be storable.** Filler turns are questions, greetings, and
  acknowledgements — the conversation a memory policy is *supposed* to discard.
  If filler became memory, the benchmark would measure noise, not retrieval.
* **Values must be distinguishable.** Answers, stale values, and distractor
  values inside one task are drawn without replacement, and a stale value is
  never a substring of the current one, so answer scoring cannot be ambiguous.
"""

from __future__ import annotations

#: Bumped whenever generated text changes: a dataset is only reproducible
#: against the generator version recorded in its manifest.
GENERATOR_VERSION = "context-rot-v1"

#: The length ladder from the plan (§14): 5k → 120k estimated tokens. Estimated
#: with the dependency-free counter (``ceil(chars / 4)``), which is what every
#: number in this repository is measured with unless a run says otherwise.
PLAN_LENGTH_LADDER: tuple[int, ...] = (5000, 10000, 20000, 40000, 80000, 120000)

#: Named length tiers. ``smoke`` is the committed dataset (small enough to
#: review in a diff); ``research`` is the plan's ladder.
TIERS: dict[str, tuple[int, ...]] = {
    "smoke": (800,),
    "quick": (2000, 4000),
    "standard": (5000, 10000, 20000, 40000),
    "research": PLAN_LENGTH_LADDER,
}
DEFAULT_TIER = "smoke"

#: Fixed default so the committed dataset is reproducible; every task also
#: records the seed it used.
DEFAULT_SEED = 20260913

#: Task categories, in plan order (A→G).
CATEGORY_SINGLE_HOP = "single_hop"
CATEGORY_MULTI_HOP = "multi_hop"
CATEGORY_TEMPORAL = "temporal"
CATEGORY_CONFLICT = "conflict"
CATEGORY_DISTRACTOR = "distractor"
CATEGORY_CROSS_SESSION = "cross_session"
CATEGORY_ABSTENTION = "abstention"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_SINGLE_HOP,
    CATEGORY_MULTI_HOP,
    CATEGORY_TEMPORAL,
    CATEGORY_CONFLICT,
    CATEGORY_DISTRACTOR,
    CATEGORY_CROSS_SESSION,
    CATEGORY_ABSTENTION,
)

CATEGORY_LABELS: dict[str, str] = {
    CATEGORY_SINGLE_HOP: "A — Single-hop retrieval",
    CATEGORY_MULTI_HOP: "B — Multi-hop retrieval",
    CATEGORY_TEMPORAL: "C — Temporal reasoning",
    CATEGORY_CONFLICT: "D — Conflict resolution",
    CATEGORY_DISTRACTOR: "E — Distractor resistance",
    CATEGORY_CROSS_SESSION: "F — Cross-session memory",
    CATEGORY_ABSTENTION: "G — Memory abstention",
}

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    CATEGORY_SINGLE_HOP: "One stored fact answers the question directly.",
    CATEGORY_MULTI_HOP: "The answer combines two facts stated far apart.",
    CATEGORY_TEMPORAL: "An older value was replaced; the newer one is correct.",
    CATEGORY_CONFLICT: "The user explicitly corrects an earlier statement.",
    CATEGORY_DISTRACTOR: "The fact is buried among many irrelevant stored facts.",
    CATEGORY_CROSS_SESSION: "A fact from an earlier session is needed later.",
    CATEGORY_ABSTENTION: "Nothing in the conversation answers the question.",
}

# ---------------------------------------------------------------------- #
# Vocabulary (drawn without replacement, so one task never has two equal
# answers or a distractingly similar pair)
# ---------------------------------------------------------------------- #

PROJECTS: tuple[str, ...] = (
    "Atlas",
    "Beacon",
    "Cobalt",
    "Delta",
    "Ember",
    "Fjord",
    "Granite",
    "Helix",
    "Ionic",
    "Juniper",
    "Kestrel",
    "Lumen",
    "Meridian",
    "Nimbus",
)

DATABASES: tuple[str, ...] = (
    "PostgreSQL 16",
    "MySQL 8",
    "SQLite 3",
    "MariaDB 11",
    "CockroachDB 23",
    "MongoDB 7",
    "TimescaleDB 2",
    "DuckDB 1",
    "Firebird 5",
    "Couchbase 7",
)

REGIONS: tuple[str, ...] = (
    "eu-west-1",
    "us-east-2",
    "ap-south-1",
    "us-west-2",
    "eu-central-1",
    "ca-central-1",
    "ap-northeast-1",
)

DEPLOY_DAYS: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
)

DEPLOY_TIMES: tuple[str, ...] = ("09:30", "13:00", "17:00", "21:45", "06:15")

OWNERS: tuple[str, ...] = ("Priya", "Marcus", "Elena", "Tomas", "Aisha", "Noah")

RETENTION_PERIODS: tuple[str, ...] = ("90 days", "180 days", "365 days", "400 days")

#: Topics the abstention category asks about. None of these words appears
#: anywhere else in the corpus, so "the conversation never mentions it" is a
#: property a test can assert rather than a hope.
PROBE_TOPICS: tuple[str, ...] = (
    "payroll vendor",
    "benefits provider",
    "legal counsel",
    "audit firm",
    "insurance carrier",
)

#: Filler turns. Every one is a question, greeting, or acknowledgement: text the
#: memory policy must reject. ``{index}`` keeps them from being byte-identical,
#: which would otherwise let a retriever dedupe the whole filler stream away and
#: distort the measured cost.
FILLER_USER_TURNS: tuple[str, ...] = (
    "How is the rollout looking today? Check {index}.",
    "Anything I should be worried about, item {index}?",
    "Can we pick this up again after lunch, round {index}?",
    "What is next on the checklist for step {index}?",
    "Let us talk about the weekend for a minute, break {index}.",
    "Any news from the on-call rotation, shift {index}?",
    "Could you summarise where we are, pass {index}?",
    "Is there anything you need from me, cycle {index}?",
)

FILLER_ASSISTANT_TURNS: tuple[str, ...] = (
    "Nothing new since the previous check, pass {index}.",
    "Still quiet here — I will flag anything unusual, pass {index}.",
    "No change on my side for round {index}.",
    "Nothing to add right now, pass {index}.",
    "All steady here, round {index}.",
)

#: Acknowledgements attached to fact turns. Kept free of statement verbs so the
#: policy does not store a second copy of a fact in the assistant's voice.
FACT_ACKS: tuple[str, ...] = (
    "Understood, thanks.",
    "Noted, thanks for the detail.",
    "Got it — thanks for flagging that.",
    "Thanks, that helps.",
    "Okay, good to know.",
)


# ---------------------------------------------------------------------- #
# Sentence templates
# ---------------------------------------------------------------------- #


def database_statement(project: str, database: str) -> str:
    return f"For the record, the production database used by Project {project} is {database}."


def database_question(project: str) -> str:
    return f"What production database does Project {project} use?"


def region_statement(project: str, region: str) -> str:
    return (
        f"For the record, Project {project} runs its production workloads in {region}."
    )


def retention_statement(project: str, period: str) -> str:
    return f"Project {project} retains audit logs for {period}."


def owner_statement(project: str, owner: str) -> str:
    return f"The delivery lead for Project {project} is {owner}."


def deploy_statement(project: str, day: str, time: str) -> str:
    return f"For Project {project}, the deployment window is {day} at {time} UTC."


def deploy_correction(project: str, day: str, time: str) -> str:
    return (
        f"Correction: the deployment window for Project {project} is {day} at {time} UTC now."
    )


def migration_statement(project: str, old_database: str, new_database: str) -> str:
    return (
        f"Project {project} has moved its production database from {old_database} "
        f"to {new_database}."
    )


def earlier_database_statement(project: str, database: str) -> str:
    return (
        f"At the start of the year, the production database for Project {project} was "
        f"{database}."
    )


def datacentre_statement(project: str, region: str) -> str:
    return f"The primary data centre for Project {project} is in {region}."


def repeated_database_statement(project: str, database: str) -> str:
    return f"Just to repeat it: Project {project} still uses {database} in production."


def distractor_fact_statement(rng: object, project: str, database: str, period: str) -> str:
    """One irrelevant-but-durable fact, used as a retrieval distractor."""

    variants = (
        database_statement(project, database),
        f"Project {project} retains its build artefacts for {period}.",
        f"For the record, the analytics warehouse for Project {project} is {database}.",
        f"Project {project} retains its release notes for {period}.",
    )
    index = rng.randrange(len(variants))  # type: ignore[attr-defined]
    return variants[index]


def multi_hop_database_statement(project: str, database: str) -> str:
    return (
        f"For the record, the telemetry service owned by Project {project} "
        f"stores its data in {database}."
    )


def multi_hop_question(database: str) -> str:
    return f"Which region hosts the project whose telemetry service stores data in {database}?"


def database_question_region(project: str) -> str:
    return f"Which region hosts the production workloads for Project {project}?"


def deploy_question(project: str) -> str:
    return f"What is the current deployment window for Project {project}?"


def migration_question(project: str) -> str:
    return f"Which production database does Project {project} use now?"


def cross_session_question(project: str) -> str:
    return f"Which region hosts the primary data centre for Project {project}?"


def abstention_question(project: str, topic: str) -> str:
    return f"Who is the {topic} for Project {project}?"


def answers_for(value: str) -> tuple[str, ...]:
    """Accepted variants for a fact value.

    Version suffixes are optional — "PostgreSQL" for "PostgreSQL 16" is a correct
    answer to a question about the database — while hyphen/space/underscore
    differences are already normalized away by the scorer.
    """

    variants = [value]
    head = value.split(" ")[0]
    if head != value:
        variants.append(head)
    return tuple(variants)


__all__ = [
    "CATEGORIES",
    "CATEGORY_ABSTENTION",
    "CATEGORY_CONFLICT",
    "CATEGORY_CROSS_SESSION",
    "CATEGORY_DESCRIPTIONS",
    "CATEGORY_DISTRACTOR",
    "CATEGORY_LABELS",
    "CATEGORY_MULTI_HOP",
    "CATEGORY_SINGLE_HOP",
    "CATEGORY_TEMPORAL",
    "DATABASES",
    "DEFAULT_SEED",
    "DEFAULT_TIER",
    "DEPLOY_DAYS",
    "DEPLOY_TIMES",
    "FACT_ACKS",
    "FILLER_ASSISTANT_TURNS",
    "FILLER_USER_TURNS",
    "GENERATOR_VERSION",
    "OWNERS",
    "PLAN_LENGTH_LADDER",
    "PROBE_TOPICS",
    "PROJECTS",
    "REGIONS",
    "RETENTION_PERIODS",
    "TIERS",
    "abstention_question",
    "answers_for",
    "cross_session_question",
    "datacentre_statement",
    "database_question",
    "database_question_region",
    "database_statement",
    "deploy_correction",
    "deploy_question",
    "deploy_statement",
    "distractor_fact_statement",
    "earlier_database_statement",
    "migration_question",
    "migration_statement",
    "multi_hop_database_statement",
    "multi_hop_question",
    "owner_statement",
    "region_statement",
    "repeated_database_statement",
    "retention_statement",
]
