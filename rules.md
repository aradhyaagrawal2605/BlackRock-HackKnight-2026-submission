# BlackRock Hackathon at IIT Delhi: Rules and Judging Criteria
Welcome to the BlackRock Hackathon at IIT Delhi 2026! We are delighted to have you participate in this exciting single-day algorithmic trading hackathon focused on building AI-powered trading agents, competing on risk-adjusted returns, and pushing the boundaries of quantitative finance.

Please read the following rules and comprehensive judging criteria to ensure a fair and rewarding hackathon experience.

## Rules 📋
- Timeline: All participants must adhere to the specified timeline and submission deadlines provided by the organizers. Data hand-off begins at 08:30 AM; the event runs from 9:00 AM – 2:30 PM.

- Team Size: Each team can have a maximum of 3 members.

- Originality: All code and resources used in the project must be original or properly attributed to the rightful owners.

- Unique Solutions: While teams are encouraged to collaborate and share knowledge, each team's solution must be unique and innovative.

- Functionality: The submitted code must be fully functional, and all 12 minimum viable requirements (see section §1.1) must be implemented as intended.

- Presentation: Qualifying teams must present a 5-minute live demo to the judging panel, showcasing their agent's strategy and results. Be prepared to answer questions about your solution. Slides are optional, but a working prototype and live terminal demo are required.

- Intellectual Property: Participants retain ownership of their work, but grant the organizers the right to showcase the agents and results for promotional purposes.

- Code of Conduct: All participants must adhere to a code of conduct that promotes a respectful and inclusive environment for everyone involved.

- Hard Constraints (Critical): Agents must never hold more than 30 tickers simultaneously (TC004) and total traded value must not exceed 30% of average portfolio value (TC005). Breaching either constraint results in immediate disqualification with a score of zero.

- LLM Quota: Each team is allocated a maximum of 60 calls to the LLM proxy endpoint for the entire session. Quota is enforced server-side. No external internet calls are permitted from the agent sandbox during the live event. This entire quota needs to be utilized wisely, as the same will be used during the live demo to judges as well. The FastAPI endpoint for the LLM will be shared on the day of the event.

- Submission: Teams must submit all four required output files (orders_log.json, portfolio_snapshots.json, llm_call_log.json, results.json) and all the code on the pre-shared Github location within 15 minutes of the event stopping at 2:30 PM.

## Judging Criteria and Metrics 🏆
- Total Score = 60% Sharpe Ratio + 20% Realized PnL + 20% Automated Test Cases (max 100 points).
- Note: Disqualified teams score 0 regardless of performance.

- Sharpe Ratio (60 points): Primary scoring metric. Measures risk-adjusted return: mean(portfolio_returns) / std(portfolio_returns). A Sharpe ≥ 3.0 earns the full 60 points. Scores below 0.5 indicate a strategy at risk of disqualification from constraint breaches.

- Realized PnL (20 points): Total portfolio value change from start ($10,000,000) to end of the 390-tick session. PnL ≥ $500,000 earns the full 20 points. Results are normalized across teams.

- Automated Test Cases (20 points): Weighted pass/fail score across 8 test cases (TC001–TC008) run by validate_solution.py against your four output files. TC004 and TC005 are disqualifying; TC007 is a bonus.

- Corporate Action Handling: How effectively does the agent detect and respond to all 7 embedded corporate action events (stock split, special dividend, earnings beat, CEO resignation, M&A rumour, regulatory fine, index rebalance)? Correct reactions are tested via TC001–TC003, TC006–TC007.

- Pitch / Demo (up to 20 bonus points): Qualifying teams present a 5-minute live demo judged on:

- Signal & Alpha Strategy (5 pts)

- Corporate Action Handling walkthrough (4 pts)

- Risk Management approach (4 pts)

- Code Quality & Architecture (4 pts)

- Retrospective & Iteration (3 pts)

## Important Notes 📌
- Final Decision: The judging panel's decision is final and cannot be contested.

- Prizes: Exciting prizes and recognition await top-performing teams, with the opportunity to showcase innovative trading solutions and connect with senior leaders in the industry.

- Resources: Participants are encouraged to leverage the provided starter kit (agent.py, optimizer.py, validate_solution.py) and all data files distributed at the start of the session.

- Creativity: Be bold, think outside the box, and dare to innovate. Originality in signal design and creative use of the 60 LLM calls are highly valued.

- Support: Our team is here to support you throughout the hackathon. For technical issues (API down, WebSocket errors), find any organizer wearing a black BlackRock T-shirt.