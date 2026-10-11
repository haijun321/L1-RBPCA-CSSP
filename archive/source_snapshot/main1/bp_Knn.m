function [mnoequl,noequl] = bp_Knn(aftest,aftrain,testlabel,trainlabel)
LabelN = size(aftest);
telabel = zeros(LabelN(1),LabelN(2),LabelN(3)); 
for s = 1:LabelN(3)
    dm = bsxfun(@minus,aftest(:,:,s),aftrain);
    dis = cumsum(cumsum(dm.^2,1),2);
    [value,in] = min(dis,[],3);
    label_matrix = trainlabel(in); 
    telabel(:,:,s) = label_matrix;
end
realabel = reshape(testlabel,[1,1,LabelN(3)]);
noequl = (telabel~=realabel);
mnoequl = mean(noequl,3);
end