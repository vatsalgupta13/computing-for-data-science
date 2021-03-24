# Money Bag Problem

# Defining global variables
initial_prize = int(input("Enter the initial money bag amount "))
wallet1_100 = int(input("Enter the number of 100 rupee bills in wallet 1 "))
wallet1_10 = int(input("Enter the number of 10 rupee bills in wallet 1 "))
wallet2_100 = int(input("Enter the number of 100 rupee bills in wallet 2 "))
wallet2_10 = int(input("Enter the number of 10 rupee bills in wallet 2 "))
cost_of_bill_selection = int(input("Enter the cost incurred for selecting a bill "))

# We keep the cost of selecting a wallet equal to the initial prize so that
# final amount won by the player is equal to the contents of the wallet
cost_of_wallet_selection = initial_prize;

def get_max_scenario(a, b, c): 
   list = [a, b, c] 
   if max(list) == a:
       return "keep the initial prize."
   elif max(list) == b:
       return "take a wallet randomly."
   else:
       return "take a bill before deciding wether or not to take a wallet."

def calc_gain(prob1, amount1, prob2, amount2):
    return ((prob1*amount1) + (prob2*amount2))

def calc_conditional(prob_bill, lr, prob_wallet):
    return ((lr/prob_bill)*prob_wallet)
 
if __name__=="__main__": 
    # defining the probabilities of selecting both the wallets (equal probability)
    p_wallet1 = p_wallet2 = 0.5
    amount_wallet1 = wallet1_100*100 + wallet1_10*10
    amount_wallet2 = wallet2_100*100 + wallet2_10*10
    take_a_wallet = calc_gain(p_wallet1, amount_wallet1 + initial_prize - cost_of_wallet_selection,
                              p_wallet2, amount_wallet2 + initial_prize - cost_of_wallet_selection)
    # total no. of bills in wallet 1 and wallet 2
    total_wallet1 = wallet1_100 + wallet1_10
    total_wallet2 = wallet2_100 + wallet2_10
    
    # calculating likelihood ratios
    lr_wallet1_10 = wallet1_10/total_wallet1
    lr_wallet1_100 = wallet1_100/total_wallet1
    lr_wallet2_10 = wallet2_10/total_wallet2
    lr_wallet2_100 = wallet2_100/total_wallet2
    
    prob_bill_10 = p_wallet1*lr_wallet1_10 + p_wallet2*lr_wallet2_100
    prob_bill_100 = p_wallet1*lr_wallet1_100 + p_wallet2*lr_wallet2_100
    keep_money_after_bill_selection = initial_prize - cost_of_bill_selection
    
    # using bayes theorem to calculate conditional probabilities
    p_wallet1_given_100 = calc_conditional(prob_bill_100, lr_wallet1_100, p_wallet1 ) 
    p_wallet1_given_10 = calc_conditional(prob_bill_10, lr_wallet1_10, p_wallet1 ) 
    p_wallet2_given_100 = calc_conditional(prob_bill_100, lr_wallet2_100, p_wallet2 ) 
    p_wallet2_given_10 = calc_conditional(prob_bill_10, lr_wallet2_10, p_wallet2 ) 
    
    take_selected_wallet_if_bill_100 = calc_gain(p_wallet1_given_100, amount_wallet1
                                                 - cost_of_bill_selection, p_wallet2_given_100, 
                                                 amount_wallet2 - cost_of_bill_selection)
    
    take_selected_wallet_if_bill_10 = calc_gain(p_wallet1_given_10, amount_wallet1 + initial_prize
                                                 - cost_of_bill_selection, p_wallet2_given_10, 
                                                 amount_wallet2 - cost_of_bill_selection)
    choose_a_bill = calc_gain(prob_bill_10, take_selected_wallet_if_bill_10,
                                                 prob_bill_100, take_selected_wallet_if_bill_100)
    
    print("Scenario 1: You keep the initial prize. Your expected gain is:", 
          initial_prize)
    print("Scenario 2: You take a wallet randomly. Your expected gain is: ", 
          take_a_wallet)
    print("Scenario 3: You choose to select a bill before deciding. Your expected gain is: ", 
          choose_a_bill)
    print("Taking into account the maximum expected gain, you should",get_max_scenario(initial_prize,take_a_wallet,choose_a_bill))