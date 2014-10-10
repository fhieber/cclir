#include "index.h"
#include "clir.h"
#include "filelib.h"

using namespace Index;
using namespace std;
int main() {
	vector<CLIR::Document> docs;
	CLIR::loadDocuments("mydocs", docs);
	Index::Index idx;
	idx.AddDocuments(docs);
	cout << idx.NumberOfTerms() << endl;
	cout << "saving idx\n";
	WriteFile out("idx.saved");
	idx.Save(*out);
	out->flush();
	sleep(5);
	cout << "loading idx2\n";
	ReadFile in("idx.saved");
	Index::Index idx2(*in);

	cout << "Index1\n" << idx << "\nIndex2\n" << idx2 << endl;
	assert(idx.NumberOfTerms() == idx2.NumberOfTerms());
	assert(idx.NumberOfDocuments() == idx2.NumberOfDocuments());
}



