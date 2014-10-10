#ifndef DISTCLIR_H_
#define DISTCLIR_H_

#include <zmq.hpp>

using namespace std;

/*
 * functions for distributed server/client CLIR
 */
namespace DISTCLIR {

#define REQUEST_TIMEOUT     10000    //  msecs, (> 1000!)

inline void receive(zmq::socket_t* socket, string& msg) {
	zmq::message_t request;
	socket->recv (&request);
	msg = string( static_cast<char*>(request.data()), request.size() );
}

inline void send(zmq::socket_t* socket, const string& msg) {
	zmq::message_t rep (msg.size());
	memcpy ((void *) rep.data(), msg.c_str(), msg.size());	
	socket->send(rep);
}

/*
 * parses a query the client has constructed
 */
inline bool parse_msg(const string& msg, string& run_id, int& k, bool& psq, string& qraw) {
	if (msg.empty())
		return false;
		
	stringstream ss;
	ss << msg;
	
	ss >> run_id;	
	if (run_id.empty())
		return false;
	ss.ignore(1,'\t');
	ss >> k;
	if (k < 0) 
		return false;
	ss.ignore(1,'\t');
	ss >> psq;
	ss.ignore(1,'\t');
	getline(ss, qraw);
	if (qraw.empty())
		return false;
	return true;
}

/*
 * constructs a query the server can parse. tries to be fail safe
 */
inline string construct_msg(const string& run_id, const int k, const bool psq, const string& query) {
	stringstream ss;
	if (run_id.empty())
		ss << "1";
	else
		ss << run_id;
	ss << '\t' << k << '\t' << psq << '\t';
	if (!query.empty())
		ss << query;
	return ss.str();
}

zmq::socket_t* setup_socket(zmq::context_t& context, const vector<string>& servers) {
	zmq::socket_t* client = new zmq::socket_t(context, ZMQ_REQ);
	int linger = 0;
	for (vector<string>::const_iterator sit=servers.begin();sit!=servers.end();++sit) {
		string con = "tcp://" + *sit;
    	cerr << "CLIENT::connecting to '" << con << "'\n";
    	client->connect( con.c_str() ); 
	}
	client->setsockopt(ZMQ_LINGER, &linger, sizeof (linger));
	return client;
}

void run_query(const string& query, zmq::socket_t* client, string& reply) {
	// send query to server(s)
	send(client, query);
	// poll socket for a reply, with timeout
	zmq::pollitem_t items[] = { { *client, 0, ZMQ_POLLIN, 0 } };
	zmq::poll (&items[0], 1, REQUEST_TIMEOUT * 1000);
	// if we got a reply, process it
	if (items[0].revents & ZMQ_POLLIN) {
		receive(client, reply);
		return;
	} else {
		cerr << "CLIENT::Error: no response from server(s)!\n";
		delete client;
		exit(1);
	}
}

inline void invalid(zmq::socket_t* socket) {
	cerr << "SERVER::REQ::invalid query\n";
	send(socket,"-1\tQ0\t-1\t1\t0.0\t1\n");
}

}

#endif
