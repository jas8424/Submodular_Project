import xml.sax
import pickle

class Handler(xml.sax.ContentHandler):  # extract all authors
    def __init__(self):
        self.CurrentData = ""  # tag's name
        self.author_dict = {}  # save all authors. The key is an author's name, the value is his id
        self.venue_dict = {}  # key: venue -> value:id
        self.graph = {}   # key: author_id -> venue_id
        self.authors = []
        self.year = ""
        self.venue = ""
        self.author = ""
        self.venue_id = 0
        self.author_id = 0
        self.depth = 0
        self.tmp_dict = {}
        self.type1 = ""

    def startElement(self, tag, attributes):
        self.CurrentData=tag
        self.depth += 1
        if self.depth == 2:
            self.flg = 1
            self.venue = ""
            self.authors = []
            self.year = ""
            self.type1=tag
        if tag == "author":
            self.author = ""


    def endElement(self, tag):
        self.CurrentData=tag
        #if tag == "artical" or tag == "proceedings" or tag == "inproceedings" or tag == "incollection":
        if self.depth == 2 and self.year != "":
            self.year=int(self.year)
            if self.year >=2019 and self.year <=2021:
                if self.venue != "":
                    tmp=list(self.venue.strip().split('/'))
                    if tmp[0]=="conf" or tmp[0]=="journals":
                        s = tmp[0]+'/'+tmp[1]
                        if s not in self.venue_dict:
                            self.venue_dict[s]=self.venue_id
                            self.venue_id += 1
                        for author in self.authors:
                            if author not in self.author_dict:
                                self.author_dict[author]=self.author_id
                                self.graph[self.author_id]=set()
                                self.author_id +=1
                            self.graph[self.author_dict[author]].add(self.venue_dict[s])

        if tag == "author":
            self.authors.append(self.author)

        self.depth -= 1


    def characters(self, content):
        if self.CurrentData == "crossref":
            self.venue += content.strip()
        elif self.CurrentData == "year":
            self.year += content.strip()
        elif self.CurrentData == "author":
            self.author += content.strip()


# set xml parser
parser = xml.sax.make_parser()
parser.setFeature(xml.sax.handler.feature_namespaces, 0)
handler1 = Handler()
parser.setContentHandler(handler1)
parser.parse('dblp.xml')

with open("venue.pkl","wb") as f:
    pickle.dump(handler1.venue_dict,f)

with open("author.pkl","wb") as f:
    pickle.dump(handler1.author_dict,f)

with open("graph.pkl","wb") as f:
    pickle.dump(handler1.graph,f)

with open('venue.txt', 'w') as f:
    for k, v in handler1.venue_dict.items():
        f.write(str(v))
        f.write(' ' + k)
        f.write('\n')
f.close()

with open('author.txt', 'w') as f:
    for k, v in handler1.author_dict.items():
        f.write(str(v))
        f.write(' ' + k)
        f.write('\n')
f.close()

with open('graph.txt', 'w') as f:
    for k, v in handler1.graph.items():
        f.write(str(k))
        f.write(':' + str(v))
        f.write('\n')
f.close()


