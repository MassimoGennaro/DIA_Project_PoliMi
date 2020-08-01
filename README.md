# DIA_Project_PoliMi
This repository contains the implementation of the project of the [Data Intelligence Application](https://www4.ceda.polimi.it/manifesti/manifesti/controller/ManifestoPublic.do?EVN_DETTAGLIO_RIGA_MANIFESTO=evento&aa=2019&k_cf=225&k_corso_la=481&k_indir=T2A&codDescr=054444&lang=IT&semestre=2&idGruppo=3925&idRiga=239696) course at Politecnico di Milano, A.Y. 2019-2020

---
## Pricing & Advertising

The goal is modeling a scenario in which a seller exploits advertising tools to attract more and more users to its website, thus increasing the number of possible buyers. The seller needs to learn simultaneously the conversion rate and the number of users the advertising tools can attract.
1. Imagine:
    1. one product to sell;
    2. three classes of users, where, for every user, we can observe the values of two binary features;
    3. the conversion rate curve of each class of users;
    4. three subcampaigns, each with a different ad, to advertise the product, and each targeting adifferent class of users;
    5. there are three abrupt phases;
    6. for every abrupt phase and for every subcampaign, the probability distribution over the dailynumber of clicks for every value of budget allocated to that subcampaign.

2. Design  a  combinatorial  bandit  algorithm  to  optimize  the  budget  allocation  over  the  three subcampaigns to maximize the total number of clicks when, for simplicity, there is only one phase. Plot the cumulative regret.

3. Design a sliding-window combinatorial bandit algorithm for the case, instead, in which there are thethree phases aforementioned. Plot the cumulative regret and compare it with the cumulative regret that a non-sliding-window algorithm would obtain.

4. Design a learning algorithm for pricing when the users that will buy the product are those that have clicked on the ads. Assume that the allocation of the budget over the three subcampaigns is fixed and there is only one phase (make this assumption also in the next steps). Plot the cumulative regret.

5. Design and run a context generation algorithm for the pricing when the budget allocated to each single subcampaign is fixed. At the end of every week, use the collected data to generate contexts and then use these contexts for the following week. Plot the cumulative regret as time increases. In the next steps, do not use the generated contexts, but use all the data together.

6. Design an optimization algorithm combining the allocation of budget and the pricing when the seller a priori knows that every subcampaign is associated with a different context and charges a different price for every context. Plot the cumulative regret when the algorithm learns both the conversion rate curves and the performance ofthe advertising subcampaigns.

7. Do the same of Step 6 under the constraint that the seller charges a unique price to all the classes of users. Plot the cumulative regret when the algorithm learns both the conversion rate curves and the performance of the advertising subcampaigns.