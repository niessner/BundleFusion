//
//  network/message.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./message.h"
# include "./core/macros.h"

namespace uplink {

//------------------------------------------------------------------------------

inline
MessageHeader::MessageHeader (Size magicSize)
: Header(magicSize)
{
    init();
}

inline
MessageHeader::MessageHeader (CString magic)
: Header(magic)
{
    init();
}

inline void
MessageHeader::init ()
{
    fields.addChunk(&kind);
    fields.addChunk(&length);
    fields.addChunk(&session);
// FIXME: fields.put(&version);
// FIXME: fields.put(&time);
// FIXME: fields.put(&checksum);
}

//------------------------------------------------------------------------------

inline
MessageSerializer::MessageSerializer (CString magic)
    : magic(magic)
    , incoming(magic)
    , outgoing(magic)
{

}

inline MessageKind
MessageSerializer::registerMessage (Message* message)
{
    assert(0 != message);
    // FIXME: Further sanity checks.

    incoming.messages.resize(std::max<size_t>(size_t(message->kind()) + 1, incoming.messages.size()));
    incoming.messages[message->kind()] = message;

    return message->kind();
}

inline
MessageSerializer::~MessageSerializer()
{
    for (int n = 0; n < incoming.messages.size(); ++n)
        delete incoming.messages[n];
}

inline Message*
MessageSerializer::readMessage (InputStream& input)
{
    // Reset.
    MessageKind kind = MessageKind_Invalid;
    incoming.header.magic = magic; // FIXME: Improve the magic situation, somehow.
    incoming.buffer.clear();

    // Read header.
    const Size headerSize = incoming.header.chunkSize();
    Byte* headerBytes = growBuffer(incoming.buffer, headerSize);
    report_zero_unless("cannot read message header", input.read(headerBytes, headerSize));

    incoming.header.fetchFrom(headerBytes);

    // Check magic.
    report_zero_unless("bad message magic", incoming.header.magic == magic);

    // Check length.
    const Size messageSize = incoming.header.length;
    report_zero_unless("bad message size", 0 <= messageSize || messageSize <= Message::MaxSize);

    // Read message.
    Byte* messageBytes = 0;

    if (0 < messageSize)
    {
        messageBytes = growBuffer(incoming.buffer, messageSize);
        report_zero_unless("cannot read message bytes", input.read(messageBytes, messageSize));
    }

    // Check kind.
    kind = incoming.header.kind;
    report_zero_unless("bad message kind", Size(kind) < incoming.messages.size());
    
    // Unpack message.
    Message* const message = incoming.messages[kind];
    assert(0 != message);
    RawBufferInputStream messageInput(messageBytes, messageSize);
    report_zero_unless("cannot unpack message", message->readFrom(messageInput));

    // Set message session id.
    message->sessionId = incoming.header.session;

    return message;
}

inline bool
MessageSerializer::writeMessage (OutputStream& output, const Message& message)
{
    // Reset packet output channel.
    outgoing.buffer.clear();

    // Prepare header.
    outgoing.header.kind = message.kind();
    outgoing.header.session = message.sessionId;

    // Get packet header size. // FIXME: Cache it in the chunk.
    const Size headerSize = outgoing.header.chunkSize();

    // Make room for packet header in the packet output buffer.
    growBuffer(outgoing.buffer, headerSize);

    // Append packet message.
    BufferOutputStream messageOutput(outgoing.buffer, headerSize);

    report_false_unless("cannot pack message", message.writeTo(messageOutput));

    // Store packet message length.
    outgoing.header.length = (MessageLength)messageOutput.count;

    // Store header at the beginning of the packet.
    Byte* headerBytes = mutableBufferBytes(outgoing.buffer); // FIXME: Remove this.
    outgoing.header.storeChunk(headerBytes);

    // Write packet.
    Byte* const packetBytes = mutableBufferBytes(outgoing.buffer);
    const Size  packetSize  = outgoing.buffer.size();

    report_false_unless("cannot write message", output.write(packetBytes, packetSize));

    return true;
}

//------------------------------------------------------------------------------

inline
MessageSerializer::Channel::Channel (CString magic)
    : header(magic)
{
    buffer.reserve(0x40000); // 256 KB
}

//------------------------------------------------------------------------------

inline
MessageInput::MessageInput (InputStream& input, MessageSerializer& messageSerializer)
    : input(input)
    , messageSerializer(messageSerializer)
{}

inline Message*
MessageInput::readMessage ()
{
    ScopedProfiledTask _(ProfilerTask_ReadMessage);

    return messageSerializer.readMessage(input);
}

//------------------------------------------------------------------------------

inline
MessageOutput::MessageOutput (OutputStream& output, MessageSerializer& messageSerializer)
    : output(output)
    , messageSerializer(messageSerializer)
{}

inline bool
MessageOutput::writeMessage (const Message& message)
{
    ScopedProfiledTask _(ProfilerTask_WriteMessage);

    return messageSerializer.writeMessage(output, message);
}

//------------------------------------------------------------------------------

inline
MessageStream::MessageStream (DuplexStream& stream, MessageSerializer& messageSerializer)
    : stream(stream)
    , messageSerializer(messageSerializer)
{

}

inline Message*
MessageStream::readMessage ()
{
    return messageSerializer.readMessage(stream);
}

inline bool
MessageStream::writeMessage (const Message& message)
{
    return messageSerializer.writeMessage(stream, message);
}

}
