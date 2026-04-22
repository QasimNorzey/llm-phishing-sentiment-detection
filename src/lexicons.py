"""Small domain lexicons for persuasion and sentiment signals.

These lexicons are intentionally lightweight so the teaching demo can run
without downloading external resources.
"""

URGENCY_WORDS = {
    "urgent", "immediately", "asap", "today", "now", "deadline", "final",
    "action", "required", "instantly", "within", "expire", "expires",
    "expiry", "soon", "alert", "attention", "promptly", "before", "tonight"
}

THREAT_WORDS = {
    "suspend", "suspended", "disable", "disabled", "locked", "block", "blocked",
    "fraud", "risk", "penalty", "failure", "failed", "warning", "breach",
    "compromised", "compromise", "unauthorized", "overdue", "delayed"
}

AUTHORITY_WORDS = {
    "security", "admin", "administrator", "it", "helpdesk", "finance", "payroll",
    "bank", "dean", "registrar", "hr", "human", "resources", "faculty",
    "office", "compliance", "support", "billing", "accounts"
}

REWARD_WORDS = {
    "reward", "bonus", "gift", "giftcard", "prize", "refund", "rebate",
    "scholarship", "benefit", "award", "voucher", "promotion"
}

CREDENTIAL_WORDS = {
    "password", "passcode", "pin", "otp", "login", "signin", "verify",
    "verification", "credentials", "authenticate", "authentication",
    "username", "code", "account"
}

ACTION_VERBS = {
    "click", "open", "confirm", "verify", "reset", "reply", "send", "submit",
    "review", "download", "unlock", "claim", "update", "login", "sign"
}

POSITIVE_WORDS = {
    "thanks", "appreciate", "success", "approved", "reward", "bonus", "great",
    "welcome", "congratulations", "benefit", "excited", "pleased"
}

NEGATIVE_WORDS = {
    "problem", "issue", "failed", "error", "risk", "warning", "urgent", "danger",
    "blocked", "suspended", "compromised", "penalty", "loss", "late", "overdue"
}

TRUST_WORDS = {
    "official", "trusted", "secure", "protected", "normal", "standard",
    "policy", "institution", "department", "verified"
}

FINANCIAL_WORDS = {
    "invoice", "payment", "bank", "salary", "bonus", "card", "refund",
    "transaction", "wire", "billing", "remittance", "tuition"
}
