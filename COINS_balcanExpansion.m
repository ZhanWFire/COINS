function [loss, gradient]= COINS_balcanExpansion(wb, k, l, value1, value2, dataU, epsilon)
%
% INPUT:
% wb        (nC,2*dim+2)    weight vector
% k         index of weight of the relevant label
% l         index of weight of the irrelevant label
% xU        (n, dim)        unlabeled inputs
% epsilon   hyperparameter in balcan's epsilon expansion theory (Balcan et. al, 2004)
%
% OUTPUT:
% loss          loss
% gradient      (nC, 2*(dim+1))     gradient

    tau = 0.8;
    [num_unlabel_data, dim] = size(dataU);
    [num_class, ~] = size(wb);
    dataU = [dataU, ones(num_unlabel_data,1)];

    v1 = value1(k,:)-value1(l,:);%(wbk1-wbl1)*dataU';
    v2 = value2(k,:)-value2(l,:);%(wbk2-wbl2)*dataU';
    y1 = 1;%sign(v1);
    y2 = 1;%sign(v2);
    ewx1 = exp(-v1.*y1);
    ewx2 = exp(-v2.*y2);
    p1 = 1./(1+ewx1);
    p2 = 1./(1+ewx2);
    c1 = COINS_ConfidenceIndicator(p1, tau);
    c2 = COINS_ConfidenceIndicator(p2, tau);

    loss = epsilon * min([c1*c2', (1-c1)*(1-c2)']) - c1*(1-c2)' - (1-c1)*c2';
    loss = loss/num_unlabel_data;

    gradient = zeros(num_class, 2*dim+2);

    if loss <= 0
        loss = 0;
    else
        pp1 = c1.*c1.*exp(-50*(p1-tau))*50;
        pp2 = c2.*c2.*exp(-50*(p2-tau))*50;

        reg1 = ewx1.*y1.*p1.*p1;
        reg2 = ewx2.*y2.*p2.*p2;
        if c1*c2' < (1-c1)*(1-c2)'
            pre1 = epsilon.*c2 - (1-c2) + c2;
            pre2 = epsilon.*c1 + c1 - (1-c1);
        else
            pre1 = -epsilon.*(1-c2) - (1-c2) + c2;
            pre2 = -epsilon.*(1-c1) + c1 - (1-c1);
        end
        grad = [ (pre1.*reg1.*pp1)*dataU, (pre2.*reg2.*pp2)*dataU ]/num_unlabel_data;
        gradient(k,:) = grad;
        gradient(l,:) = -grad;
    end

end
