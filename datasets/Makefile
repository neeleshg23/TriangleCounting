SUBDIRS = ak2010 asia_osm belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 delaunay_n24 hollywood-2009 kron_g500-logn21 road_central road_usa roadNet-CA soc-twitter-2010  

.PHONY: $(GRAPHS)

all: recurse

recurse: 
	for subdir in $(SUBDIRS); do $(MAKE) -C $$subdir; done

$(GRAPHS):
	$(MAKE) -C $@
