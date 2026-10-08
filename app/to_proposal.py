"""Turn the verified extraction into the object we hand to window.loadProposal().

Rules:
  * A key is only present if the notes gave us the value. Unknown means omitted
    (the generator keeps its default; sending "" would wipe it).
  * The amounts are written to strategy.keyFigures (the tool reads this one)
    AND goalAssessment.keyFigures (where proposal-schema.md and the sample
    outputs put them).
  * The client's situation is written to `needs` (as documented) AND to
    goalAssessment.rationale (the tool prints this one).

This is the only place that knows the shape of the proposal, so if Old Mutual
changes the contract, this is the only file to edit.
"""

from decimal import Decimal


def _plain(value):
    return format(Decimal(str(value)).normalize(), "f")


def to_proposal(clean):
    p = {}

    if "client_name" in clean:
        p["clientName"] = clean["client_name"]

    p["introGreetTo"] = "advisor" if clean.get("addressed_to") == "advisor" else "client"

    if "financial_advisor_name" in clean:
        p["financialPlanner"] = clean["financial_advisor_name"]

    if "model" in clean:
        p["applyModel"] = clean["model"]

    situation = clean.get("client_situation")
    if situation:
        p["needs"] = situation

    goal = {}
    if "client_type" in clean:
        goal["clientType"] = clean["client_type"]
    if "mandate_type" in clean:
        goal["mandateType"] = clean["mandate_type"]
    if "currency" in clean:
        goal["currency"] = clean["currency"]
    if "horizon" in clean:
        goal["investmentHorizon"] = clean["horizon"]
    if situation:
        goal["rationale"] = situation

    key_figures = {}
    if "investment_amount" in clean:
        key_figures["totalInvestment"] = _plain(clean["investment_amount"])
    if "income_amount" in clean:
        key_figures["incomeValue"] = _plain(clean["income_amount"])
    if key_figures:
        goal["keyFigures"] = dict(key_figures)
    if goal:
        p["goalAssessment"] = goal

    objective = {}
    if "target_return" in clean:
        objective["targetReturn"] = clean["target_return"]
    if "benchmark" in clean:
        objective["benchmark"] = clean["benchmark"]
    if "risk_profile" in clean:
        objective["riskProfile"] = clean["risk_profile"]
    if "horizon" in clean:
        objective["investmentHorizon"] = clean["horizon"]
    if objective:
        p["objective"] = objective

    strategy = {}
    if clean.get("allocation"):
        strategy["allocation"] = [
            {"class": r["class"], "key": r["key"],
             "percent": int(r["percent"]) if r["percent"] == r["percent"].to_integral() else float(r["percent"])}
            for r in clean["allocation"]
        ]
    if key_figures:
        strategy["keyFigures"] = dict(key_figures)
    if "currency" in clean:
        strategy["currency"] = clean["currency"]
    if strategy:
        p["strategy"] = strategy

    if "adviser_fee_percent" in clean:
        p["fees"] = {"advisorFee": format(Decimal(str(clean["adviser_fee_percent"])), ".2f")}

    if clean.get("is_replacement"):
        replacement = {"isReplacement": True}
        if clean.get("replacement_details"):
            replacement["details"] = clean["replacement_details"]
        p["replacements"] = replacement

    return p