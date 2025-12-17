"""Dataset loading for speculative decoding experiments.

Loads prompts from five domains to study how acceptance rates vary across
task types:

- **code**: HumanEval programming problems
- **math**: GSM8K grade-school math
- **instruct**: Alpaca-cleaned instruction following
- **creative**: Creative / general writing from the Pile
- **long**: Long-document summarisation from GovReport (SCROLLS)

Falls back to synthetic prompts when datasets are unavailable.
"""

from __future__ import annotations

import warnings
from typing import Dict, List

from config import PROMPTS_PER_DOMAIN


# ────────────────────────────────────────────────────────────
# Individual loaders (each returns a list of prompt strings)
# ────────────────────────────────────────────────────────────


def _load_humaneval(n: int) -> List[str]:
    """Load HumanEval code-completion prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("openai_humaneval", split="test", trust_remote_code=False)
        return [row["prompt"] for row in ds.select(range(min(n, len(ds))))]
    except Exception as e:
        warnings.warn(f"HumanEval unavailable ({e}). Using fallback code prompts.")
        return _fallback_code_prompts(n)


def _load_gsm8k(n: int) -> List[str]:
    """Load GSM8K math reasoning prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="train", trust_remote_code=False)
        return [
            f"Solve: {row['question']}"
            for row in ds.select(range(min(n, len(ds))))
        ]
    except Exception as e:
        warnings.warn(f"GSM8K unavailable ({e}). Using fallback math prompts.")
        return _fallback_math_prompts(n)


def _load_alpaca(n: int) -> List[str]:
    """Load Alpaca-cleaned instruction prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("yahma/alpaca-cleaned", split="train", trust_remote_code=False)
        prompts = []
        for row in ds:
            inst = row.get("instruction", "")
            if inst and len(inst) > 20:
                prompts.append(inst)
            if len(prompts) >= n:
                break
        return prompts[:n]
    except Exception as e:
        warnings.warn(f"Alpaca unavailable ({e}). Using fallback instruct prompts.")
        return _fallback_instruct_prompts(n)


def _load_creative(n: int) -> List[str]:
    """Load creative writing prompts from the Pile (or fallback)."""
    try:
        from datasets import load_dataset

        # The Pile is huge; use streaming to grab a few examples
        ds = load_dataset(
            "monology/pile-uncopyrighted",
            split="train",
            streaming=True,
            trust_remote_code=False,
        )
        prompts = []
        for row in ds:
            text = row.get("text", "")
            # Take the first ~200 chars as a prompt
            if len(text) > 100:
                prompts.append(text[:500])
            if len(prompts) >= n:
                break
        if len(prompts) >= n:
            return prompts[:n]
    except Exception:
        pass

    warnings.warn("Pile dataset unavailable. Using fallback creative prompts.")
    return _fallback_creative_prompts(n)


def _load_govreport(n: int) -> List[str]:
    """Load GovReport long-document prompts from SCROLLS."""
    try:
        from datasets import load_dataset

        ds = load_dataset("tau/scrolls", "gov_report", split="test", trust_remote_code=False)
        prompts = []
        for row in ds.select(range(min(n, len(ds)))):
            text = row.get("input", "")
            # Truncate to first ~512 tokens worth of text (~2048 chars)
            prompts.append(text[:2048])
        return prompts[:n]
    except Exception as e:
        warnings.warn(f"GovReport unavailable ({e}). Using fallback long prompts.")
        return _fallback_long_prompts(n)


# ────────────────────────────────────────────────────────────
# Fallback prompts (always available, no network needed)
# ────────────────────────────────────────────────────────────


def _fallback_code_prompts(n: int) -> List[str]:
    """Synthetic code completion prompts."""
    bases = [
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        "def merge_sort(arr):\n    \"\"\"Sort an array using merge sort.\"\"\"\n",
        "def binary_search(arr, target):\n    \"\"\"Find target in sorted array.\"\"\"\n",
        "class LinkedList:\n    \"\"\"Singly linked list implementation.\"\"\"\n    def __init__(self):\n",
        "def is_palindrome(s):\n    \"\"\"Check if string is a palindrome.\"\"\"\n",
        "def matrix_multiply(A, B):\n    \"\"\"Multiply two matrices.\"\"\"\n",
        "def dijkstra(graph, source):\n    \"\"\"Find shortest paths using Dijkstra.\"\"\"\n",
        "def lru_cache(capacity):\n    \"\"\"Implement an LRU cache.\"\"\"\n",
        "def topological_sort(graph):\n    \"\"\"Perform topological sort on DAG.\"\"\"\n",
        "def knapsack(weights, values, capacity):\n    \"\"\"Solve 0/1 knapsack.\"\"\"\n",
        "def quicksort(arr):\n    \"\"\"Sort using quicksort.\"\"\"\n",
        "def bfs(graph, start):\n    \"\"\"Breadth-first search.\"\"\"\n",
        "def dfs(graph, start):\n    \"\"\"Depth-first search.\"\"\"\n",
        "class Stack:\n    \"\"\"Stack data structure.\"\"\"\n    def __init__(self):\n",
        "def gcd(a, b):\n    \"\"\"Compute greatest common divisor.\"\"\"\n",
        "class MinHeap:\n    \"\"\"Min-heap implementation.\"\"\"\n    def __init__(self):\n",
        "def longest_common_subsequence(s1, s2):\n    \"\"\"Find LCS of two strings.\"\"\"\n",
        "def count_inversions(arr):\n    \"\"\"Count inversions in array.\"\"\"\n",
        "def trie_insert(root, word):\n    \"\"\"Insert word into trie.\"\"\"\n",
        "class Graph:\n    \"\"\"Adjacency list graph.\"\"\"\n    def __init__(self):\n",
    ]
    return (bases * ((n // len(bases)) + 1))[:n]


def _fallback_math_prompts(n: int) -> List[str]:
    """Synthetic math reasoning prompts."""
    bases = [
        "Solve: If a train travels at 60 mph for 3 hours, how far does it go?",
        "Solve: What is the sum of the first 100 positive integers?",
        "Solve: A rectangle has length 12 and width 5. What is its area?",
        "Solve: If 3x + 7 = 22, what is x?",
        "Solve: A store has a 25% off sale. If a shirt costs $40, what is the sale price?",
        "Solve: How many prime numbers are between 1 and 50?",
        "Solve: If a circle has radius 7, what is its circumference?",
        "Solve: A baker made 144 cookies and packed them in boxes of 12. How many boxes?",
        "Solve: What is 15% of 200?",
        "Solve: If the ratio of boys to girls is 3:5 and there are 40 students, how many boys?",
        "Solve: A car depreciates 20% per year. After 3 years, what fraction of value remains?",
        "Solve: What is the probability of rolling a sum of 7 with two dice?",
        "Solve: If f(x) = 2x^2 + 3x - 5, what is f(4)?",
        "Solve: A triangle has sides 3, 4, and 5. What is its area?",
        "Solve: How many ways can you arrange the letters in MISSISSIPPI?",
        "Solve: What is the integral of x^2 from 0 to 3?",
        "Solve: A ball is dropped from 100 meters. How long until it hits the ground?",
        "Solve: If log base 2 of x equals 5, what is x?",
        "Solve: What is the determinant of [[1,2],[3,4]]?",
        "Solve: A coin is flipped 10 times. What is P(exactly 3 heads)?",
    ]
    return (bases * ((n // len(bases)) + 1))[:n]


def _fallback_instruct_prompts(n: int) -> List[str]:
    """Synthetic instruction-following prompts."""
    bases = [
        "Explain the theory of relativity in simple terms.",
        "Write a haiku about the ocean.",
        "List five benefits of regular exercise.",
        "Describe the process of photosynthesis step by step.",
        "Compare and contrast Python and JavaScript.",
        "What are the main causes of climate change?",
        "Explain how a blockchain works to a 10-year-old.",
        "Give me a recipe for chocolate chip cookies.",
        "What is the difference between machine learning and deep learning?",
        "Summarize the plot of Romeo and Juliet.",
        "Explain quantum computing in three sentences.",
        "What are the pros and cons of remote work?",
        "Describe the water cycle in detail.",
        "What are the key principles of object-oriented programming?",
        "Explain the concept of supply and demand.",
        "List the planets in our solar system in order.",
        "What is the difference between DNA and RNA?",
        "Describe how the internet works.",
        "What are the main features of a democratic government?",
        "Explain how vaccines work.",
    ]
    return (bases * ((n // len(bases)) + 1))[:n]


def _fallback_creative_prompts(n: int) -> List[str]:
    """Synthetic creative writing prompts."""
    bases = [
        "The old lighthouse keeper hadn't seen a ship in forty years. But tonight, through the fog, something was approaching — and it wasn't a ship.",
        "She found the letter tucked inside a library book, written in handwriting that looked exactly like her own, dated fifty years in the future.",
        "The last human on Earth sat alone in a room. There was a knock on the door.",
        "Every night at exactly 3:17 AM, the radio in the attic turns on by itself, playing a song that hasn't been written yet.",
        "The city appeared on no map, yet everyone who visited remembered it as if they'd lived there their entire lives.",
        "He inherited his grandmother's antique shop, along with a peculiar rule: never open the blue cabinet on the second floor.",
        "The astronaut opened her visor on the alien planet and took a deep breath of air that smelled exactly like fresh-baked bread.",
        "Two strangers on a park bench discover they've been having the exact same dream for the past seven nights.",
        "The painting in the museum changed every night. The security guard was the only one who noticed.",
        "A detective investigates a murder that hasn't happened yet, using evidence that keeps appearing on her desk.",
        "The forest had been silent for three days. Then the trees began to sing.",
        "She woke up fluent in a language that no linguist could identify.",
        "The bridge between the two kingdoms was exactly one step wide, and only appeared during thunderstorms.",
        "He could hear colors. The sunset sounded like a symphony.",
        "The village had no shadows. Not the buildings, not the people, not even the sundial.",
        "A librarian discovers that returning overdue books causes small changes in history.",
        "The mirror showed everything truthfully — except the person standing in front of it.",
        "Rain fell upward in this town, and nobody thought it was strange.",
        "She was the only person who could see the threads connecting people to their futures.",
        "The clock tower had been broken for a century. When it finally struck midnight, time itself paused.",
    ]
    return (bases * ((n // len(bases)) + 1))[:n]


def _fallback_long_prompts(n: int) -> List[str]:
    """Synthetic long-document prompts (government report style).

    Contains 15 distinct prompts spanning different policy areas to avoid
    cycling duplicates when n > len(bases).
    """
    bases = [
        # 1 — Infrastructure / Transportation
        "The following is a government report on infrastructure spending in fiscal year 2023. "
        "The Department of Transportation allocated $45 billion to highway maintenance and "
        "construction projects across 48 states. Key findings indicate that bridge repair "
        "costs have increased by 23% since 2020, while rural road maintenance has been "
        "deferred in 35% of counties due to budget constraints. The report recommends "
        "a phased approach to infrastructure modernization with emphasis on climate "
        "resilience and sustainable materials. Summarize the key findings.",

        # 2 — Environmental Protection / Air Quality
        "Annual report of the Environmental Protection Agency on air quality standards "
        "compliance across metropolitan areas. In 2023, 78% of monitored cities met "
        "federal ozone standards, up from 71% in 2022. However, particulate matter "
        "exceedances increased in the western United States due to wildfire activity. "
        "The report details new emission reduction targets for the industrial sector "
        "and proposes updated monitoring protocols. Provide a summary.",

        # 3 — Healthcare Expenditures
        "Congressional Budget Office analysis of healthcare expenditures for fiscal "
        "year 2024. Total federal healthcare spending reached $1.8 trillion, representing "
        "28% of the federal budget. Medicare and Medicaid accounted for 75% of these "
        "expenditures. The report projects that without reform, healthcare spending will "
        "consume 35% of the federal budget by 2035. Summarize the projections.",

        # 4 — Defense / Military Readiness
        "Department of Defense readiness assessment for fiscal year 2024. The report "
        "evaluates force structure across all branches, noting that 62% of Army brigade "
        "combat teams achieved full readiness ratings compared to 58% the prior year. "
        "Naval vessel maintenance backlogs grew by 14%, with three carrier strike groups "
        "delayed in dry-dock refits. The assessment recommends $12 billion in supplemental "
        "funding for depot-level maintenance and accelerated recruitment programs to "
        "address a 41,000-person shortfall in active-duty end strength. Summarize the "
        "readiness gaps and funding recommendations.",

        # 5 — Education
        "Government Accountability Office report on K-12 education funding equity across "
        "rural and urban school districts. Analysis of 14,000 districts found that per-pupil "
        "spending in the lowest-income quartile averaged $11,200 compared to $16,800 in the "
        "highest-income quartile — a 33% gap that has widened since 2019. Title I allocations "
        "offset only 18% of this disparity. The report recommends revised allocation formulas "
        "weighted by poverty concentration and English-learner populations, along with "
        "targeted grants for teacher retention in underserved areas. What are the main "
        "findings and policy recommendations?",

        # 6 — Energy
        "Department of Energy annual review of the national electricity grid and renewable "
        "energy integration. In 2023, renewable sources (wind, solar, hydro) supplied 22% "
        "of total U.S. electricity generation, up from 20% in 2022. Grid interconnection "
        "queues contained over 2,000 GW of proposed projects, but only 14% historically "
        "reach commercial operation. Transmission bottlenecks in ERCOT and MISO regions "
        "caused $3.7 billion in congestion costs. The review proposes permitting reform "
        "and $8 billion in transmission expansion. Summarize the challenges and proposals.",

        # 7 — Agriculture
        "USDA Economic Research Service report on farm income and agricultural trade "
        "for calendar year 2023. Net farm income fell 23% to $136 billion, driven by "
        "lower commodity prices for corn (−19%), soybeans (−15%), and wheat (−12%). "
        "Agricultural exports totaled $178 billion, a 6% decline from the prior year, "
        "with China remaining the largest single-country destination at $26 billion. "
        "The report examines the impact of drought conditions across the Southern Plains "
        "on livestock feed costs and crop insurance payouts. Provide an overview of the "
        "income trends and trade dynamics.",

        # 8 — Labor / Workforce
        "Bureau of Labor Statistics analysis of workforce participation and wage growth "
        "trends from 2020 through 2024. The labor force participation rate recovered to "
        "62.7%, still 0.6 percentage points below the pre-pandemic level. Real wage growth "
        "turned positive in Q3 2023 after 26 months of negative readings. The gig economy "
        "accounted for an estimated 36% of the workforce, though measurement challenges "
        "persist. Occupational projections show healthcare support and technology roles "
        "growing fastest through 2032. Summarize the participation and wage findings.",

        # 9 — Housing
        "Department of Housing and Urban Development report on the national housing "
        "shortage and affordability crisis. The estimated deficit stands at 4.5 million "
        "units as of mid-2024, concentrated in metropolitan areas with population growth "
        "exceeding 1.5% annually. Median home prices reached 5.8 times median household "
        "income nationally, the highest ratio on record. Rental vacancy rates fell to "
        "5.6%, and 49% of renters are cost-burdened (spending more than 30% of income on "
        "housing). The report outlines proposed zoning reforms, tax incentives for "
        "multifamily construction, and expanded voucher funding. What solutions does the "
        "report propose?",

        # 10 — Veterans Affairs
        "Inspector General audit of the Department of Veterans Affairs healthcare delivery "
        "system. Wait times for primary care averaged 26 days nationally, with 12 VA "
        "medical centers exceeding 45 days. Community care referrals under the MISSION Act "
        "grew 34% year-over-year, straining the $22 billion annual community care budget. "
        "Electronic health record modernization (Oracle Cerner) deployment was paused at "
        "five sites pending resolution of 149 patient safety tickets. The audit recommends "
        "staffing increases of 23,000 clinical positions and revised scheduling standards. "
        "Summarize the audit findings and recommendations.",

        # 11 — Cybersecurity
        "Cybersecurity and Infrastructure Security Agency annual threat assessment for "
        "fiscal year 2024. State-sponsored intrusion attempts targeting critical "
        "infrastructure increased 47% over the prior year, with energy and water sectors "
        "most affected. Ransomware incidents reported to CISA rose to 2,825, causing an "
        "estimated $1.4 billion in recovery costs. The assessment highlights emerging "
        "risks from AI-generated phishing and deepfake social engineering campaigns. "
        "Recommended actions include mandatory multi-factor authentication for federal "
        "contractors and an $800 million grant program for state and local government "
        "cyber resilience. Outline the key threats and recommended actions.",

        # 12 — Trade / Commerce
        "U.S. International Trade Commission economic analysis of tariff impacts on "
        "domestic manufacturing and consumer prices. Tariffs enacted between 2018 and 2023 "
        "on steel, aluminum, and Chinese goods generated $89 billion in cumulative revenue "
        "but increased consumer costs by an estimated $1,200 per household annually. "
        "Domestic steel production rose 8% while downstream manufacturers reported 11% "
        "higher input costs. The analysis examines trade diversion effects, with imports "
        "from Vietnam, India, and Mexico increasing 62%, 38%, and 27% respectively in "
        "tariff-affected categories. Provide a balanced summary of the costs and benefits.",

        # 13 — Public Health / Pandemic Preparedness
        "HHS Office of the Assistant Secretary for Preparedness and Response review of "
        "pandemic readiness infrastructure following lessons from COVID-19. The Strategic "
        "National Stockpile holds 85% of target quantities for critical medical "
        "countermeasures, up from 47% in 2020. Domestic N-95 respirator manufacturing "
        "capacity reached 1.5 billion units annually. However, the public health workforce "
        "declined by 15% since 2020 due to burnout and funding lapses. The review proposes "
        "a $6.1 billion multi-year investment in genomic surveillance networks, wastewater "
        "monitoring, and local health department capacity. Summarize the readiness status "
        "and investment proposals.",

        # 14 — Immigration
        "Congressional Research Service report on immigration enforcement and processing "
        "backlogs. Immigration court pending cases reached 3.3 million at the end of FY2024, "
        "with an average wait time of 4.3 years. Customs and Border Protection reported "
        "2.04 million encounters at the southwest border, a 12% decrease from FY2023. "
        "Asylum officer staffing increased 20% under the Asylum Processing Rule but remains "
        "insufficient to meet demand. The report analyzes the fiscal impact of work "
        "authorization delays, estimating $4.2 billion in foregone tax revenue annually. "
        "What are the key backlogs and their economic impacts?",

        # 15 — Water Infrastructure
        "EPA assessment of drinking water and wastewater infrastructure needs nationwide. "
        "An estimated $625 billion in investment is required over the next 20 years to "
        "maintain and upgrade water systems. Lead service line replacements are underway "
        "in 9,300 of the 12,000 systems known to contain them. Rural water systems serving "
        "fewer than 10,000 people face the most acute funding gaps, with 23% reporting "
        "violations of Safe Drinking Water Act standards. PFAS contamination has been "
        "detected in water supplies serving 110 million people, and the new maximum "
        "contaminant levels will require $12 billion in treatment upgrades. Describe the "
        "scale of investment needed and the most pressing challenges.",
    ]
    return (bases * ((n // len(bases)) + 1))[:n]


# ────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────


def load_datasets(prompts_per_domain: int = PROMPTS_PER_DOMAIN) -> Dict[str, List[str]]:
    """Load prompts from all five domains.

    Args:
        prompts_per_domain: Number of prompts to load from each domain.

    Returns:
        Dictionary mapping domain name → list of prompt strings.
    """
    print(f"Loading {prompts_per_domain} prompts per domain...")

    datasets: Dict[str, List[str]] = {}

    loaders = {
        "code": _load_humaneval,
        "math": _load_gsm8k,
        "instruct": _load_alpaca,
        "creative": _load_creative,
        "long": _load_govreport,
    }

    for domain_name, loader_fn in loaders.items():
        print(f"  Loading {domain_name}...", end=" ", flush=True)
        try:
            prompts = loader_fn(prompts_per_domain)
            datasets[domain_name] = prompts
            print(f"OK ({len(prompts)} prompts)")
        except Exception as exc:
            print(f"FAILED ({exc})")
            datasets[domain_name] = []

    total = sum(len(v) for v in datasets.values())
    print(f"Loaded {total} prompts across {len(datasets)} domains.\n")
    return datasets
