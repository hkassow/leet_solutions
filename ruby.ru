# #problem 322 Coin Change
# DP 
# if your solution is slower than o(n) can you sort array?
# infinity in ruby uses Float::INFINITY
def coin_change(coins, amount)
    
    dp = Array.new(amount + 1, Float::INFINITY)
    dp[0] = 0
    coins.sort
    coins.each{|coin| 
        
        j = coin 
        while j <= amount 
            dp[j] = [dp[j], dp[j-coin] + 1].min
            j += 1
        end
    }
    dp[amount] == Float::INFINITY ? -1: dp[amount]
end