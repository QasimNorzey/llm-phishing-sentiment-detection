from __future__ import annotations

from pathlib import Path
import random

import pandas as pd

SEED = 42
random.seed(SEED)

SERVICES = [
    "Microsoft 365", "PayPal", "Outlook", "University Portal", "Dropbox", "Banking Portal", "Payroll System"
]
SUSPICIOUS_DOMAINS = [
    "secure-auth-check.com", "verify-center.net", "payroll-sync.info", "account-review.co", "m365-alerts.io"
]
OFFICIAL_DOMAINS = [
    "portal.university.edu", "hr.university.edu", "it.university.edu", "finance.university.edu", "library.university.edu"
]
NAMES = ["Dr. Ahmed", "Sara", "Omar", "Mona", "IT Support", "Finance Office", "HR Team", "Registrar"]

PHISHING_TEMPLATES = [
    "Urgent: your {service} account will be suspended today. Verify your password now at https://{domain}/login to avoid immediate deactivation.",
    "Security alert for {service}. We detected unusual activity and must confirm your credentials within 30 minutes. Click https://{domain}/verify.",
    "Payroll correction notice: a payment failure was detected. Submit your banking details immediately using https://{domain}/payroll to release your salary.",
    "Confidential request from {name}: buy 5 gift cards before noon and reply with the codes. Keep this urgent task private.",
    "Mailbox quota warning. Your email will stop receiving messages tonight unless you login and revalidate your account through https://{domain}/storage.",
    "You are eligible for a scholarship refund. Confirm your university login, card number, and OTP today at https://{domain}/refund.",
    "Action required: your MFA reset failed. To restore access now, verify your sign-in code and password at https://{domain}/mfa.",
    "Finance approval pending. Open the attached invoice and send the updated wire details immediately so the transfer is not delayed.",
    "Tax reimbursement available for staff. Claim your reward by confirming your account credentials now at https://{domain}/claim.",
    "Final warning: unauthorized sign-in from abroad. Reply with your username and verification code immediately to block the attack.",
]

BENIGN_TEMPLATES = [
    "Reminder: the research seminar has moved to 3 PM in Room 214. Please verify your attendance in the official portal at https://{domain}/events before Friday.",
    "IT maintenance notice: the email system will restart tonight. No password sharing is required; use the normal sign-in page tomorrow morning.",
    "HR reminder from {name}: please update your emergency contact in the employee portal by the end of the month.",
    "Finance Office update: March reimbursement forms are available at https://{domain}/forms. Review the policy document next week if needed.",
    "Library alert: your borrowed book is due soon. Renew it in the official campus app or contact the circulation desk for help.",
    "Course coordinator notice: the assignment deadline has been extended. Submit through Blackboard as usual and do not email your password to anyone.",
    "Payroll message: payslips are now available on the internal portal. If something looks wrong, call payroll directly instead of sending banking data by email.",
    "Registrar announcement: students can confirm enrollment status through the official university website this week.",
    "Department meeting reminder from {name}: bring the printed agenda and project updates to tomorrow's session.",
    "Security awareness bulletin: never click unknown links or share OTP codes. Report suspicious messages to IT Support immediately.",
]


def generate_examples(label: int, templates: list[str], n: int) -> list[dict[str, object]]:
    rows = []
    for idx in range(n):
        template = random.choice(templates)
        text = template.format(
            service=random.choice(SERVICES),
            domain=random.choice(SUSPICIOUS_DOMAINS if label == 1 else OFFICIAL_DOMAINS),
            name=random.choice(NAMES),
        )
        rows.append({"text": text, "label": label})
    return rows


def main() -> None:
    phishing = generate_examples(1, PHISHING_TEMPLATES, 80)
    benign = generate_examples(0, BENIGN_TEMPLATES, 80)
    df = pd.DataFrame(phishing + benign).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    output = Path("data/demo_emails.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
