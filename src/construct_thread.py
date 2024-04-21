import pandas as pd

class ConsrtuctThread:
    def __init__(self, dataframe, id, comment_level = -1):
        self.dataframe = dataframe
        self.id = id
        self.comment_level = comment_level
        self.thread = self.get_thread()

    def get_thread(self):
        '''reconstruct the Reddit thread (containing all posts if comment_level=-1) from the given reddit_name/reddit_link_id/reddit_parent_id. 
       If comment_level=1 then the reconstruction is restricted only to level 1 comments''' 
        id_name = self.id
        if(self.comment_level == -1):
            # Reconstructs the entire post with the given id with all the comments in a somewhat unstructured manner. A sorting is done with respect to reddit_created_utc to keep the temporal flow of information
            return pd.concat([self.dataframe[self.dataframe['reddit_name']==id_name], self.dataframe[self.dataframe['reddit_link_id']==id_name]]).sort_values(by=['reddit_created_utc'])
        if(self.comment_level == 1):
            # Reconstructs the entire post with the given id keeping only level 1 comments. A sorting is done with respect to reddit_created_utc to keep the temporal flow of information
            arr = []
            arr.append(self.dataframe[self.dataframe['reddit_name']==id_name])
            arr.append(self.dataframe[self.dataframe['reddit_parent_id']==id_name])
            return pd.concat(arr).sort_values(by=['reddit_created_utc'])
        
    def get_url(self):
        return "https://www.reddit.com/r/" + self.get_thread().iloc[0].reddit_subreddit + "/comments/" + self.get_thread().iloc[0].reddit_id
    
    def get_title(self):
        return self.thread.iloc[0].reddit_title
    
    def get_conversation(self, user=False):
        if user:
            temp = self.thread
            return temp[["reddit_author", "reddit_text"]].set_index("reddit_author").to_dict()['reddit_text']
        if not user:
            # title = temp[temp['aware_post_type'] == 'submission'].iloc[0].reddit_title
            return list(self.thread.reddit_text)
        
    def get_author_list(self):
        return list(set(self.thread["reddit_author"]))
