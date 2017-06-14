function [confidenceIndicator] = COINS_ConfidenceIndicator(confidence, tau)
%     tau = 0.8;
    confidenceIndicator = 1./(1+exp(-50*(confidence-tau)));
end
