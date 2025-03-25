import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
  Query,
} from "@nestjs/common";
import { ChatService } from "./chat.service";

@Controller("chat")
export class ChatController {
  constructor(private readonly chatService: ChatService) {}

  @Post()
  create(@Body() data) {
    return this.chatService.create(data);
  }

  @Post("find-chat")
  findChat(@Body() data) {
    return this.chatService.findChat(data);
  }
  
  @Get("find-doctor-chat-list/:doctorId")
  findDoctorChatList(@Param("doctorId") doctorId: string) {
    return this.chatService.findDoctorChatList(doctorId);
  }

  @Get("find-user-chat-list/:userId")
  findUserChatList(@Param("userId") userId: string) {
    return this.chatService.findUserChatList(userId);
  }
}
