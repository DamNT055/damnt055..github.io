from collections import OrderedDict
import operator
class Solution:
    def topKFrequent(nums, k):
        ans = dict()
        for x in nums:
            ans[x] = ans.get(x, 0) + 1 
        ans = dict(sorted(ans.items(),
                           key=operator.itemgetter(1),
                           reverse=True)[:2])
        return ans.keys()
Solution.topKFrequent([1,1,1,2,2,3,3,4], 1)