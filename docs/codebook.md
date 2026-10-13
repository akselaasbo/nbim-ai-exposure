# AI Exposure Codebook — v1

You classify one listed company at a time. Answer one question:

> **Over the next five years (2026–2031), what is the dominant way artificial intelligence
> affects this company's earnings?**

Judge the company as it is today, based on what you know. Do not guess about future
strategy changes that have not been announced.

## Materiality test
AI only counts if it is plausibly **one of the top three drivers of the company's
five-year investment case** — something a serious equity analyst would put on the first
page of a report. Almost every company "uses AI". That alone is not enough.
If AI does not pass this test, the answer is LIMITED.

## Categories — apply in this order, the first rule that fits wins

### 1. ENABLER
The company sells the inputs AI runs on, and this is a top-three driver.
Includes: semiconductors and chip equipment, memory, networking, cloud and hyperscale
compute, foundation model developers that sell model access, data-centre owners and
builders, and power, cooling and electrical equipment where data-centre demand is a
top-three driver.
Examples: Nvidia, ASML, TSMC, Microsoft (Azure), Schneider Electric.

### 2. DISRUPTED
A material part of the company's core revenue could be substituted or heavily
price-compressed by AI, and this risk is a top-three driver.
Includes: IT outsourcing and business-process outsourcing, call centres, translation,
stock content, routine content and information services, test prep and ed-tech,
seat-based software where AI agents can replace the users, staffing for routine office work.
Examples: Teleperformance, Infosys, Shutterstock.
If the company is also adopting AI, DISRUPTED still wins when substitution risk is the bigger driver.

### 3. BENEFICIARY
AI materially improves the company's product, pricing power or cost base, the expected
net effect is positive, and the company is not mainly selling AI infrastructure.
Includes sellers of AI applications and companies where AI is central to the product.
Examples: Meta Platforms (AI-driven ad targeting), Palantir, Spotify.

### 4. LIMITED
AI is not a top-three driver of the five-year investment case.
This is the expected answer for most utilities, consumer staples, materials, real estate,
banks, insurers, industrials and energy companies, unless a rule above clearly applies.

## Edge-case rules
| Situation | Decision |
|---|---|
| Hyperscalers (Microsoft, Alphabet, Amazon) | ENABLER — rule order puts selling compute first |
| Utilities and power producers | LIMITED, unless the company itself presents data-centre demand as a main growth driver → ENABLER |
| Strong upside *and* credible substitution risk (e.g. some software, information services, consultancies) | DISRUPTED if substitution of the core product is seriously debated; set confidence to low or medium |
| Banks and insurers using AI internally | LIMITED, unless AI is clearly a top-three driver |
| Holding companies and conglomerates | Classify by the largest segment by revenue |
| Company you do not recognise | Infer from name, country and industry, set `knows_company` = false and confidence = low |

## Output fields
- `category`: ENABLER, DISRUPTED, BENEFICIARY or LIMITED
- `confidence`:
  - `high` — an informed analyst would agree without much debate
  - `medium` — reasonable people could choose another category
  - `low` — you are largely guessing
- `knows_company`: true if you know what this specific company does; false if you are
  inferring from the name, country and industry. If false, confidence must be `low`.
- `rationale`: one sentence, at most 25 words, naming the concrete mechanism
  (what is sold, what is substituted, or why AI is not material).
