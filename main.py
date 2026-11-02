'''
Here we do the simplest first step:
1. Generate the number cards from the game
2. Generate a function which just keeps "hitting"
3. Tracks when it busts (both how many cards and the total value it busted at)
'''


import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

cards = [int(x) for x in [0]+list(np.repeat(np.arange(1, 13), np.arange(1, 13)))]


print("Inital deck of cards: ", cards)

def hit(deck):
    # First, we need to randomly select a card from the deck
    sample_idx = np.random.randint(0, len(deck))

    # Now we need to get the value of that card
    card_value = deck[sample_idx]

    # Now we need to remove that card from the deck
    deck = np.delete(deck, sample_idx)

    return card_value, deck

# -------------------------------HERE--------------------------------
n_simulations = 5_00_000
cumulative_values = []
final_hands = []

for _ in tqdm(range(n_simulations)):
    # First, we need to create a copy of the deck so that we can modify it without affecting the original deck
    deck = cards.copy()
    current_hand = []
    current_sums = [0]

    while True:
        card_value, deck = hit(deck)
        
        # We bust if we have 2 of the same card in our hand
        if card_value in current_hand:
            num_cards = len(current_hand)
            total_value = sum(current_hand)
            break

        current_hand.append(card_value)
        current_sums.append(current_sums[-1] + card_value)

    final_hands.append(current_hand)
    cumulative_values.append(current_sums)


    

# Convert the lists to numpy arrays for easier manipulation
#cumulative_values = np.array(cumulative_values)


# Calculate the number of cards had when busted and the total value for each simulation
num_cards_busted = np.array([len(hand)-1 for hand in cumulative_values])
total_value_busted = np.array([hand[-1] for hand in cumulative_values])

print("Best hand: ", final_hands[np.argmax(total_value_busted)])


# Now we will visualize the results of our simulations

# We first need to create a histogram of the number of cards busted
max_num_cards = np.max(num_cards_busted)
plt.hist(num_cards_busted, bins=np.arange(1, max_num_cards+1)-0.5, density=True)
plt.xticks(np.arange(1, max_num_cards+1))
plt.xlabel("Number of Cards before Busted")
plt.ylabel("Frequency")
plt.title("Histogram of Number of Cards before Busted")
plt.show()


# Now we will create a histogram of the total value busted
# To make the histogram more readable, we will only label every other bin
max_total_value = max(total_value_busted)
plt.hist(total_value_busted, bins=np.arange(1, max_total_value+1)-0.5, density=True)
plt.xticks(np.arange(1, max_total_value, 2))
plt.xlabel("Total Value before Busted")
plt.ylabel("Frequency")
plt.title("Histogram of Total Value before Busted")
plt.show()

# Now we will create a plot showing your probability of busting as a function of the number of cards in your hand
bust_probabilities = []
for i in range(1, max_num_cards+1):
    # Calculate the probability of busting with i cards in hand
    # We can do this by counting the number of times we busted with i cards in hand and dividing it by the total number of times we had i cards in hand
    num_busts_with_i_cards = np.sum(num_cards_busted == i)
    num_times_with_i_cards = np.sum(num_cards_busted >= i)
    
    bust_probability = num_busts_with_i_cards / num_times_with_i_cards if num_times_with_i_cards > 0 else None

    bust_probabilities.append(bust_probability)

# We can now plot the probability of busting as a function of the number of cards in hand
plt.plot(range(1, max_num_cards+1), bust_probabilities, "o-")
plt.xlabel("Number of Cards in Hand")
plt.ylabel("Probability of Busting")
plt.title("Probability of Busting vs. Number of Cards in Hand")
plt.show()

# Now we will create a plot showing your probability of busting as a function of the total value in your hand
bust_probabilities_total_value = []
for i in tqdm(range(1, max_total_value+1)):
    # Calculate the probability of busting with a total value of i in hand
    # We can do this by counting the number of times we busted with a total value of i in hand and dividing it by the total number of times we had a total value of i in hand
    num_busts_with_total_value_i = np.sum(total_value_busted == i)

    hands_with_total_value_i = [hand for hand in cumulative_values if i in hand]
    num_times_with_total_value_i = len(hands_with_total_value_i)
    bust_probability_total_value = num_busts_with_total_value_i / num_times_with_total_value_i if num_times_with_total_value_i > 0 else 0

    bust_probabilities_total_value.append(bust_probability_total_value)

# We can now plot the probability of busting as a function of the total value in your hand
plt.plot(range(1, max_total_value+1), bust_probabilities_total_value, "o-")
plt.xlabel("Total Value in Hand")
plt.ylabel("Probability of Busting")
plt.title("Probability of Busting vs. Total Value in Hand")
plt.show()


# Save probability of busting as a function of the total value in your hand
value2bust_prob = {
    zip(np.arange(1,max_total_value+1), bust_probabilities_total_value)
}

np.savez("bust_probabilities.npz", value2bust_prob)



# 2D heatmap
bust_prob2d = -np.ones((max_num_cards, max_total_value))

# For each hand value we need to count how many times we had it at each number of cards
# And how many times it was the amount we busted on at each number of cards

# First lets count how many times they occur

occurence_counts = np.zeros((max_num_cards+1, max_total_value+1))
last_counts = np.zeros((max_num_cards+1, max_total_value+1))

for cum_vals in cumulative_values:
    for idx,val in enumerate(cum_vals):
        occurence_counts[idx,val] += 1

    last_counts[idx,val] += 1

    

bust_prob2d = last_counts / occurence_counts
plt.imshow(bust_prob2d)
plt.colorbar()
plt.show()

np.save("bust_prob2d.npy", bust_prob2d)




# Strategy Testing
# Let's say your strategy is to keep hitting up until you get some number N
# How would that play out?

# To get the scores you would have gotten you need to count two things:
# 1) The number of times you would bust before getting to N
# 2) Your score after hitting or passing N

import bisect

cutoff_scores = np.arange(1, 79+1)

expected_values = []
prob_bustings = []

for cutoff_score in cutoff_scores:
    num_times_busted = np.sum(total_value_busted < cutoff_score)
    scores_not_busted = []
    for cum_vals in cumulative_values:
        if cum_vals[-1] >= cutoff_score:
            idx = bisect.bisect_left(cum_vals, cutoff_score)
            scores_not_busted.append(cum_vals[idx])

    prob_busting = num_times_busted / (num_times_busted + len(scores_not_busted))
    expected_value = np.mean(scores_not_busted + num_times_busted*[0])

    expected_values.append(expected_value)
    prob_bustings.append(prob_busting)

expected_values = np.array(expected_values)
prob_bustings = np.array(prob_bustings)

fig,ax = plt.subplots(2, sharex=True)
ax[0].plot(cutoff_scores, expected_values, "o-")
ax[1].plot(cutoff_scores, prob_bustings, "o-")

ax[0].set_title("Expected Value by Cutoff Score")
ax[1].set_title("Probability of Busting by Cutoff Score")
plt.show()

