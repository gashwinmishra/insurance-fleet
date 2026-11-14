from flows.orchestrator import ClaimOrchestrator


class AutoClaimFlow:
    """
    Domain-specific Auto Claim Flow orchestrated via ClaimOrchestrator.
    Maintains backwards-compatible attributes (`visual_agent`, `adjudicator`, `logger`)
    and `.run(image_input)` interface.
    """

    def __init__(self):
        self.orchestrator = ClaimOrchestrator(default_claim_type="auto")
        self.visual_agent = self.orchestrator.visual_agent
        self.adjudicator = self.orchestrator.adjudicator
        self.logger = self.orchestrator.logger

    def run(self, image_input):
        print("--- Starting Auto Claim Flow ---")
        return self.orchestrator.run(image_input, claim_type="auto")

