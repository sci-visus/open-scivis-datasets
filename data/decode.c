#include <stdlib.h>

#include <zfp.h>


size_t
decode(unsigned char *buffer, size_t buffer_size, float *array, int width, int height, int depth)
{
	zfp_type type = zfp_type_float;
	zfp_field *field = zfp_field_3d(array, type, width, height, depth);
	zfp_stream *zfp = zfp_stream_open(NULL);
	zfp_stream_set_precision(zfp, 8); // MUST AGREE WITH SERVER

	bitstream *stream = stream_open(buffer, buffer_size);
	zfp_stream_set_bit_stream(zfp, stream);
	size_t result = zfp_decompress(zfp, field);

	zfp_field_free(field);
	zfp_stream_close(zfp);
	stream_close(stream);

	return result;
}
