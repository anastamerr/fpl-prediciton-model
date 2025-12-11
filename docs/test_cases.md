# FPL RAG System - Test Cases & Expected Results

## Retrieval Method Guide

| Method | Best For | How It Works |
|--------|----------|--------------|
| **Baseline (Cypher)** | Exact queries, rankings, stats, filters | Direct database queries with ORDER BY |
| **Embeddings** | Semantic similarity, "players like X", vague queries | Vector similarity search |
| **Hybrid** | Best of both - combines structured + semantic | Fuses results from both methods |

---

## Test Case 1: Top Players by Position

### Query: "Who are the top forwards this season?"

| Method | Expected Top Results | Why |
|--------|---------------------|-----|
| **Baseline** | Haaland, Salah, Kane (by total_points DESC) | Exact ranking by points |
| **Embeddings** | Haaland, Wilson, Watkins, Toney | Semantic match to "top forward" descriptors |
| **Hybrid** | Haaland (appears in both), plus mix | Best coverage |

**Expected Intent:** `position_search`
**Expected Entities:** `positions: ["FWD"]`

---

## Test Case 2: Player Comparison

### Query: "Compare Haaland vs Kane for 2022-23"

| Method | Expected Result |
|--------|-----------------|
| **Baseline** | Side-by-side stats: points, goals, assists, ICT for both players |
| **Embeddings** | May return similar players to both (less useful here) |
| **Hybrid** | Baseline stats + related players |

**Expected Intent:** `comparison_query`
**Expected Entities:** `players: ["Erling Haaland", "Harry Kane"], seasons: ["2022-23"]`

---

## Test Case 3: Budget Team Recommendation

### Query: "Recommend a squad with budget of 100m"

| Method | Expected Result |
|--------|-----------------|
| **Baseline** | 15 players fitting 3-4-3 formation under 100m budget |
| **Embeddings** | Top performers (not budget-aware) |
| **Hybrid** | Budget-constrained picks enhanced with form data |

**Expected Intent:** `team_recommendation`
**Expected Entities:** `numerical_values: {budget: 100}`

---

## Test Case 4: Fixture Analysis

### Query: "Show me fixtures for Liverpool in gameweek 10"

| Method | Expected Result |
|--------|-----------------|
| **Baseline** | Liverpool's GW10 fixture(s) with opponent, home/away |
| **Embeddings** | Not ideal for this query type |

**Expected Intent:** `fixture_analysis`
**Expected Entities:** `teams: ["Liverpool"], gameweeks: [10]`

---

## Test Case 5: Player Performance

### Query: "How did Salah perform in gameweek 15?"

| Method | Expected Result |
|--------|-----------------|
| **Baseline** | Salah's GW15 stats: points, goals, assists, minutes, bonus |
| **Embeddings** | Similar high-performing midfielders |

**Expected Intent:** `player_performance`
**Expected Entities:** `players: ["Mohamed Salah"], gameweeks: [15]`

---

## Test Case 6: Semantic/Similarity Query (Embeddings Excel Here)

### Query: "Find players similar to Kevin De Bruyne"

| Method | Expected Result |
|--------|-----------------|
| **Baseline** | Limited - no similarity template |
| **Embeddings** | Creative midfielders with high assists: Ødegaard, Maddison, etc. |
| **Hybrid** | Embedding results dominate |

**Expected Intent:** `general_question`

---

## Test Case 7: Vague/Natural Language Query

### Query: "Who should I captain this week?"

| Method | Expected Result |
|--------|-----------------|
| **Baseline** | May struggle without specific entities |
| **Embeddings** | Top performers with "must-have", "premium" descriptors |
| **Hybrid** | Best approach for vague queries |

---

## Test Case 8: Historical Query

### Query: "Best XI from 2021-22 season"

| Method | Expected Result |
|--------|-----------------|
| **Baseline** | Formation-based best players by position for that season |

**Expected Intent:** `historical_query`
**Expected Entities:** `seasons: ["2021-22"]`

---

## Test Case 9: Statistics Query

### Query: "Players with most goals in 2022-23"

| Method | Expected Result |
|--------|-----------------|
| **Baseline** | Top scorers ordered by goals_scored DESC |
| **Embeddings** | Players with "prolific goal scorer", "league's top scorer" |

**Expected Intent:** `statistics_query`
**Expected Entities:** `statistics: ["goals_scored"], seasons: ["2022-23"]`

---

## Test Case 10: Position + Budget Filter

### Query: "Best midfielders under 8m"

| Method | Expected Result |
|--------|-----------------|
| **Baseline** | Top midfielders by points (price data not available) |
| **Embeddings** | High-return midfielders (ignores price) |

**Expected Intent:** `position_search`
**Expected Entities:** `positions: ["MID"]`

---

## Test Case 11: Form Analysis

### Query: "Which players are in form right now?"

| Method | Expected Result |
|--------|-----------------|
| **Baseline** | Players with highest points in last 5 gameweeks |
| **Embeddings** | "Outstanding performer", "strong performer" matches |

**Expected Intent:** `form_analysis`

---

## Test Case 12: Team Analysis

### Query: "Which teams have the best defense?"

| Method | Expected Result |
|--------|-----------------|
| **Baseline** | Teams ordered by fewest goals conceded |

**Expected Intent:** `team_analysis`

---

## Quick Test Script

Run these queries in the Streamlit app and verify:

```
1. "Who are the top forwards this season?"
   → Should return: Haaland, Kane, Wilson, Watkins, Toney

2. "Compare Salah and Son"
   → Should return: Side-by-side stats for both players

3. "Best defenders under 5.5m"
   → Should return: Budget DEF options with good points

4. "Find players like Bruno Fernandes"
   → Should return: Creative MIDs with high assists (use Embeddings)

5. "Recommend a team with 95m budget"
   → Should return: 15 players in valid formation

6. "Liverpool fixtures gameweek 20"
   → Should return: Liverpool's GW20 match details

7. "Top scorers 2022-23"
   → Should return: Haaland, Kane, etc. ordered by goals

8. "Who should I pick for my wildcard?"
   → Vague query - Hybrid works best
```

---

## Method Selection Guide

| Query Type | Recommended Method |
|------------|-------------------|
| Rankings (top X, best, most) | Baseline or Hybrid |
| Comparisons | Baseline |
| Budget constraints | Baseline |
| Fixtures/Gameweeks | Baseline |
| "Similar to X" | Embeddings |
| Vague/conversational | Hybrid |
| Historical stats | Baseline |
| Form/recent performance | Baseline or Hybrid |

---

## Debugging Tips

1. **Check Intent**: Expand "Parsed entities" in the UI to see detected intent
2. **Check Entities**: Verify players/teams/positions were extracted correctly
3. **Check Cypher**: Expand "Cypher Queries Executed" to see the actual query
4. **Check Embedding Hits**: Expand "Knowledge Graph Retrieved Context" to see raw results
5. **If Embeddings fail**: Run `python scripts/check_embeddings.py` to verify storage
