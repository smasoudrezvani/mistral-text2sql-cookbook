# Why text-to-SQL answers fail: one real example per error type

System: Mistral Large 4, approach (c) execute-and-retry, BIRD mini-dev (2026 release), 500 questions, 65.0% correct. 50 of the 175 wrong answers were labelled one by one (`results/error_labels/retry_mistral-large-4_all_sample50.json`).

| Category | Count (of 50) |
|---|---|
| Dirty data | 21 |
| Wrong filter (misread question) | 9 |
| Wrong column | 6 |
| Wrong filter value | 4 |
| Wrong aggregation | 4 |
| Format only (the answer is actually right) | 4 |
| Wrong join | 1 |
| Ambiguous question | 1 |

## Dirty data (21 of 50)

**Question** (q348, database `superhero`): I want to profile the cohort of heroes whose Intelligence score matches Abomination's Intelligence score. What is their average measured weight in kg? Please round the average to two decimals.

**Hint:** Intelligence is one of the six attribute scores a hero has (Intelligence, Strength, Speed, Durability, Power, Combat)

**What went wrong:** In this database a weight of 0 means "unknown". The model averaged the zeros in as real weights, so the average dropped from 103.6 to 82.14 kg.

**Model's SQL:**
```sql
SELECT ROUND(AVG(s.weight_kg), 2) AS average_weight_kg
FROM superhero s
JOIN hero_attribute ha ON ha.hero_id = s.id
JOIN attribute a ON a.id = ha.attribute_id
WHERE a.attribute_name = 'Intelligence'
  AND s.weight_kg IS NOT NULL
  AND ha.attribute_value IN (
      SELECT ha2.attribute_value
      FROM superhero s2
      JOIN hero_attribute ha2 ON ha2.hero_id = s2.id
      JOIN attribute a2 ON a2.id = ha2.attribute_id
      WHERE s2.superhero_name = 'Abomination'
        AND a2.attribute_name = 'Intelligence'
  )
```

**Model's answer:**
```
82.14
```

**Gold SQL:**
```sql
SELECT ROUND(AVG(CASE WHEN s.weight_kg > 0 THEN s.weight_kg END), 2) AS average_weight_kg FROM superhero s JOIN hero_attribute ha ON ha.hero_id = s.id JOIN attribute a ON a.id = ha.attribute_id WHERE a.attribute_name = 'Intelligence' AND ha.attribute_value = (SELECT abom_ha.attribute_value FROM superhero abom JOIN hero_attribute abom_ha ON abom_ha.hero_id = abom.id JOIN attribute abom_a ON abom_a.id = abom_ha.attribute_id WHERE abom.superhero_name = 'Abomination' AND abom_a.attribute_name = 'Intelligence')
```

**Gold answer:**
```
103.6
```

## Wrong filter (misread question) (9 of 50)

**Question** (q390, database `thrombosis_prediction`): Among patients in the thrombosis registry born between 1930 and 1940 inclusive who had laboratory tests, what percentage were inpatients? Please give me just the percentage rounded to two decimal places.

**Hint:** An inpatient is a patient who was admitted to the hospital; an outpatient is a patient who was followed at the outpatient clinic.

**What went wrong:** "Patients in the thrombosis registry" means everyone in this database (it is a thrombosis registry; the database is even called `thrombosis_prediction`). The model saw the word "thrombosis", saw a `Thrombosis` column, and connected them: it added `AND e.Thrombosis = 1`, so it only counted patients who actually had thrombosis and the answer became 100% instead of 35.9%. Language models tend to use every important word of the question somewhere in the SQL, even words that only describe the database. The same mistake happened 3 times in the 50 labelled failures (q390, q430, q439), always with "thrombosis registry/cohort". A possible fix is a prompt rule: words that describe the database or the whole cohort are not filters.

**Model's SQL:**
```sql
SELECT ROUND(
         100.0 * SUM(CASE WHEN p.Admission = '+' THEN 1 ELSE 0 END)
         / COUNT(*),
         2
       ) AS percentage
FROM Patient AS p
WHERE p.Birthday >= '1930-01-01'
  AND p.Birthday <= '1940-12-31'
  AND EXISTS (
        SELECT 1
        FROM Examination AS e
        WHERE e.ID = p.ID
          AND e.Thrombosis = 1
      )
  AND EXISTS (
        SELECT 1
        FROM Laboratory AS l
        WHERE l.ID = p.ID
      )
```

**Model's answer:**
```
100.0
```

**Gold SQL:**
```sql
SELECT ROUND(CAST(SUM(CASE WHEN Admission = '+' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*), 2) AS percentage FROM Patient WHERE STRFTIME('%Y', Birthday) BETWEEN '1930' AND '1940' AND ID IN (SELECT ID FROM Laboratory)
```

**Gold answer:**
```
35.9
```

## Wrong column (6 of 50)

**Question** (q145, database `codebase_community`): I'm profiling our most active contributors of 2012. Among the users who created more than ten posts in 2012 (judged by each post's creation date), what is their average number of up-votes cast and their average age? Please give me a single row with both averages, each rounded to two decimal places.

**Hint:** Up-vote: a vote a user cast in favor of a post.

**What went wrong:** "Average number of up-votes cast" is stored in users.UpVotes. The model counted rows in the votes table instead, where the user id is almost always empty, so it got 0.

**Model's SQL:**
```sql
WITH active_2012 AS (
  SELECT p.OwnerUserId AS user_id
  FROM posts p
  WHERE p.OwnerUserId IS NOT NULL
    AND p.CreaionDate >= '2012-01-01'
    AND p.CreaionDate < '2013-01-01'
  GROUP BY p.OwnerUserId
  HAVING COUNT(*) > 10
),
cohort AS (
  SELECT u.Id AS user_id,
         u.Age AS age,
         (SELECT COUNT(*) FROM votes v WHERE v.UserId = u.Id AND v.VoteTypeId = 2) AS upvotes_cast
  FROM users u
  JOIN active_2012 a ON a.user_id = u.Id
)
SELECT ROUND(AVG(upvotes_cast), 2) AS avg_upvotes_cast,
       ROUND(AVG(age), 2) AS avg_age
FROM cohort
```

**Model's answer:**
```
0.0 | 36.14
```

**Gold SQL:**
```sql
WITH active AS (SELECT OwnerUserId FROM posts WHERE CreaionDate >= '2012-01-01' AND CreaionDate < '2013-01-01' GROUP BY OwnerUserId HAVING COUNT(*) > 10) SELECT ROUND(AVG(u.UpVotes), 2) AS avg_upvotes, ROUND(AVG(u.Age), 2) AS avg_age FROM active AS a JOIN users AS u ON u.Id = a.OwnerUserId
```

**Gold answer:**
```
505.63 | 36.14
```

## Wrong filter value (4 of 50)

**Question** (q423, database `thrombosis_prediction`): I'd like to know how many patients on record have both an abnormal anti-SSA antibody result and a positive proteinuria finding at some point in their record. Abnormal anti-SSA means any recorded test result other than negative or zero. Proteinuria counts as positive if any urine test on file returned a result other than negative or none. The two findings don't need to come from the same visit – I'm looking across each patient's entire record. Please just give me the number of such patients.

**Hint:** anti-SSA: anti-SSA antibody laboratory test; proteinuria: protein found in the urine

**What went wrong:** The model wrote '-' for a negative anti-SSA result, but this column stores the word 'negative', so negatives were counted as abnormal and the count doubled.

**Model's SQL:**
```sql
SELECT COUNT(DISTINCT p.ID)
FROM Patient AS p
WHERE EXISTS (
    SELECT 1
    FROM Laboratory AS l
    WHERE l.ID = p.ID
      AND l.SSA IS NOT NULL
      AND l.SSA NOT IN ('-', '0')
)
AND EXISTS (
    SELECT 1
    FROM Laboratory AS l
    WHERE l.ID = p.ID
      AND l.`U-PRO` IS NOT NULL
      AND l.`U-PRO` NOT IN ('-', '0')
)
```

**Model's answer:**
```
31
```

**Gold SQL:**
```sql
SELECT COUNT(*) AS patient_count FROM Patient AS T1 WHERE T1.ID IN (SELECT ID FROM Laboratory WHERE SSA IS NOT NULL AND SSA NOT IN ('negative', '0')) AND T1.ID IN (SELECT ID FROM Laboratory WHERE `U-PRO` IS NOT NULL AND `U-PRO` NOT IN ('-', '0'))
```

**Gold answer:**
```
15
```

## Wrong aggregation (4 of 50)

**Question** (q293, database `formula_1`): For my retrospective on the 1962 British Grand Prix – the race named 'British Grand Prix' in the 1962 season – how many drivers withdrew from that race? I just need a single whole number.

**Hint:** A driver who withdrew from a race is recorded with the status 'Withdrew'.

**What went wrong:** One driver appears twice in the results table for this race. COUNT(*) counts rows; the question asks for drivers, so COUNT(DISTINCT driverId) is needed.

**Model's SQL:**
```sql
SELECT COUNT(*)
FROM results
JOIN races ON results.raceId = races.raceId
JOIN status ON results.statusId = status.statusId
WHERE races.year = 1962
  AND races.name = 'British Grand Prix'
  AND status.status = 'Withdrew'
```

**Model's answer:**
```
5
```

**Gold SQL:**
```sql
SELECT COUNT(DISTINCT x.driverId) AS num_withdrawn_drivers FROM results AS x JOIN races AS r ON r.raceId = x.raceId JOIN status AS s ON s.statusId = x.statusId WHERE r.name = 'British Grand Prix' AND r.year = 1962 AND s.status = 'Withdrew'
```

**Gold answer:**
```
4
```

## Format only (the answer is actually right) (4 of 50)

**Question** (q86, database `card_games`): Among the Commander-leadable cards that the Card Kingdom catalog lists for both a foil printing and a regular non-foil printing – cards listed for only one of the two printings don't count – I want the single most recent card frame year (the latest calendar year among the frame versions), and the count of cards in that group whose frame version is that year. Please return just that one row: the frame year and its card count.

**Hint:** A card 'may lead a Commander deck' if it can be the deck's commander – the one card that leads the deck and sets its color identity – not merely be included among the deck's other cards.

**What went wrong:** Same year and count, but the model returned the year as the number 2015 and the gold as the text '2015'. Exact matching (ours and BIRD's) treats them as different.

**Model's SQL:**
```sql
WITH eligible AS (
  SELECT CAST(frameVersion AS INTEGER) AS frame_year
  FROM cards
  WHERE frameVersion IS NOT NULL
    AND CAST(frameVersion AS INTEGER) > 0
    AND cardKingdomId IS NOT NULL
    AND cardKingdomFoilId IS NOT NULL
    AND LOWER(REPLACE(REPLACE(REPLACE(COALESCE(leadershipSkills, ''), ' ', ''), char(34), ''), char(39), '')) LIKE '%commander:true%'
)
SELECT frame_year,
       COUNT(*) AS card_count
FROM eligible
WHERE frame_year = (SELECT MAX(frame_year) FROM eligible)
GROUP BY frame_year
```

**Model's answer:**
```
2015 | 820
```

**Gold SQL:**
```sql
WITH cmd AS (SELECT frameVersion, substr(leadershipSkills, instr(leadershipSkills, 'commander')) AS tail FROM cards WHERE cardKingdomId IS NOT NULL AND cardKingdomFoilId IS NOT NULL) SELECT frameVersion AS frame_year, COUNT(*) AS card_count FROM cmd WHERE substr(tail, length('commander') + length('True'), length('True')) = 'True' GROUP BY frameVersion ORDER BY CAST(frameVersion AS INTEGER) DESC LIMIT 1
```

**Gold answer:**
```
2015 | 820
```

## Wrong join (1 of 50)

**Question** (q25, database `california_schools`): I'm working on a story about campuses Los Angeles County opened during 2007 and the SAT participation they drew. What was the average number of SAT test takers per school across Los Angeles County schools that opened during 2007 and are run by a school district? Please give me that single number, rounded to two decimals. A few things to keep in mind: a county office of education counts as the school district for the schools it runs; "opened during 2007" means the opening date falls in calendar year 2007; and a school with no SAT record counts as zero test takers.

**Hint:** SAT test takers: the number of test takers recorded for the school.

**What went wrong:** An INNER JOIN to the frpm table silently dropped schools that have no frpm row, and the district type was taken from that table instead of schools.DOCType.

**Model's SQL:**
```sql
SELECT ROUND(AVG(COALESCE(t.NumTstTakr, 0)), 2)
FROM schools AS sch
JOIN frpm AS f
  ON f.CDSCode = sch.CDSCode
LEFT JOIN satscores AS t
  ON t.cds = sch.CDSCode
 AND t.rtype = 'S'
WHERE sch.County = 'Los Angeles'
  AND sch.OpenDate >= '2007-01-01'
  AND sch.OpenDate < '2008-01-01'
  AND f.`District Type` IN (
    'Elementary School District',
    'High School District',
    'Unified School District',
    'County Office of Education (COE)'
  )
```

**Model's answer:**
```
22.83
```

**Gold SQL:**
```sql
SELECT ROUND(SUM(COALESCE(sat.NumTstTakr, 0)) * 1.0 / COUNT(*), 2) AS avg_sat_test_takers FROM schools AS s LEFT JOIN satscores AS sat ON sat.cds = s.CDSCode WHERE s.County = 'Los Angeles' AND strftime('%Y', s.OpenDate) = '2007' AND (s.DOCType LIKE '%school district%' OR s.DOCType LIKE '%county office of education%')
```

**Gold answer:**
```
14.57
```

## Ambiguous question (1 of 50)

**Question** (q438, database `thrombosis_prediction`): I'm reviewing patients with an elevated anti-ribonuclear protein result. Among patients who ever had an anti-ribonuclear protein result of 16 or higher, I want to follow up the three whose liver enzymes stayed in the normal range. The normal range for GPT is below 60. For each patient, take their highest GPT value among their laboratory tests that fall in the normal range. Rank patients from that highest value down; if two patients share the same value, the one with the smaller patient ID comes first. Please return exactly three patients, one row per patient, with only each patient's Birthday.

**Hint:** GPT stands for glutamic pyruvic transaminase, a liver enzyme; anti-ribonuclear protein is an autoantibody measured in laboratory tests.

**What went wrong:** The question says "the three whose liver enzymes **stayed** in the normal range" and then "take their highest GPT value **among their laboratory tests that fall in the normal range**". This can be read in two ways. **Reading A** (the model's, and most people's): "stayed in the normal range" means the GPT was *never* 60 or above, so any patient with an abnormal test is excluded. **Reading B** (the gold's): nobody is excluded; just ignore the abnormal tests and rank patients by their highest normal one. The table below shows why the answers differ: the gold's top patient had 30 abnormal GPT tests (up to 227), which hardly "stayed in the normal range". So the question is ambiguous, and the gold is arguably questionable.

**Model's SQL:**
```sql
WITH rnp_high AS (
  SELECT DISTINCT l.ID
  FROM Laboratory l
  WHERE CAST(l.RNP AS REAL) >= 16
),
max_normal_gpt AS (
  SELECT l.ID, MAX(l.GPT) AS max_gpt
  FROM Laboratory l
  WHERE l.GPT < 60
  GROUP BY l.ID
)
SELECT p.Birthday
FROM Patient p
JOIN rnp_high r ON r.ID = p.ID
JOIN max_normal_gpt m ON m.ID = p.ID
WHERE NOT EXISTS (
  SELECT 1
  FROM Laboratory l2
  WHERE l2.ID = p.ID
    AND l2.GPT >= 60
)
ORDER BY m.max_gpt DESC, p.ID ASC
LIMIT 3
```

**Model's answer:**
```
1958-10-23
1969-11-30
1962-01-18
```

**Gold SQL:**
```sql
SELECT T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.GPT < 60 AND T1.ID IN (SELECT ID FROM Laboratory WHERE CAST(RNP AS REAL) >= 16) GROUP BY T1.ID ORDER BY MAX(T2.GPT) DESC, T1.ID ASC LIMIT 3
```

**Gold answer:**
```
1949-05-11
1958-10-23
1963-12-13
```

**The two readings, with the real data** (patients with RNP >= 16, ranked by their highest *normal* GPT test; generated from the database):

| Patient | Birthday | Highest normal GPT | Highest GPT ever | Abnormal GPT tests | Reading A keeps? |
|---|---|---|---|---|---|
| 5186147 | 1949-05-11 | 59 | 227 | 30 | no |
| 2320830 | 1958-10-23 | 57 | 57 | 0 | yes |
| 5184284 | 1963-12-13 | 57 | 80 | 3 | no |
| 5122312 | 1969-11-30 | 56 | 56 | 0 | yes |
| 1650222 | 1947-03-17 | 44 | 93 | 1 | no |
| 912147 | 1975-02-17 | 42 | 219 | 5 | no |

Reading B (gold) takes the first three rows. Reading A (model) skips every patient with an abnormal test, so it takes 1958-10-23, 1969-11-30 and the next patient with none.

## How to fix each kind of error

| Error | Fix | How |
|---|---|---|
| Dirty data (21/50) | **Add BIRD's column descriptions to the schema** | BIRD's description CSVs already say e.g. *"If the weight_kg is NULL or 0, it means the weight of the superhero is missing"* and *"Thrombosis: 0 = negative, 1-3 = degree"*. The plain prompt only had CREATE TABLE + 3 example rows. |
| Dirty data | **Data profile per column** (automatic, no LLM) | For each column compute: share of NULL / 0 / '' / '-', numbers stored as text, values that look like lists ("a, b"). Add one short note per risky column, e.g. `weight_kg: 31% are 0 (likely missing)`. |
| Dirty data | **System-prompt rules** | "Treat 0, '' and '-' in measurement columns as missing unless the question says otherwise. CAST text numbers before MIN/MAX/comparisons. Fields may hold comma-separated lists." |
| Wrong filter (misread) | **System-prompt rule + self-check** | "The database itself is the cohort/registry; don't add filters the question doesn't ask for." Ask the model to list each condition it used and point to the words in the question that require it. |
| Wrong column | **Column descriptions** (same as above) | Descriptions say which column holds what (e.g. `users.UpVotes` = up-votes cast). |
| Wrong filter value | **Show the distinct values of small columns** | For columns with <= 20 distinct values, list them in the schema (`Thrombosis: 0, 1, 2, 3`; `SSA: 'negative', '0', '1', '4', ...`). Value retrieval (extra 2) only helps when the words match. |
| Wrong aggregation | **System-prompt rule + self-check** | "When counting entities (drivers, patients), use COUNT(DISTINCT id); check whether joins create duplicates." |
| Format only | **System-prompt rule** and/or a lenient scorer as a footnote | "Return exactly the requested columns; don't combine or re-type values." A lenient comparison would add only 7 of 500 for Large 4 (c). |
| Wrong join | **System-prompt rule** | "Use LEFT JOIN when the question says to include all rows (with zeros)." |
| Ambiguous | Not fixable by prompting | Report it; ideally the question should be clarified. |

**We tested the first two rows together as "column notes"** (BIRD descriptions + an automatic
data profile, `--notes` in `scripts/run_eval.py`), with BIRD's hints still in the prompt:

| Model | Plain prompt | + column notes |
|---|---|---|
| Ministral 3 8B (500 questions) | 10.6% | 16.4% |
| Mistral Small 4 (500 questions) | 18.6% | 30.8% |
| Mistral Large 4 (200-question subset) | 61.5% | 72.0% |

All three gains are statistically significant (paired McNemar test, p <= 0.01). The prompt rules
for misread filters and aggregations were not tested.

To inspect any question yourself (free, no API calls):
`uv run python scripts/show_case.py <question_id>`
