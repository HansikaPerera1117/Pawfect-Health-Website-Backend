import { Injectable } from "@nestjs/common";
import { PrismaService } from "src/prisma/prisma.service";

@Injectable()
export class ChatService {
  constructor(private readonly prismaSerice: PrismaService) {}
  create(data: any) {
    return this.prismaSerice.chat.create({
      data: {
        isDoctorMessage: data.isDoctorMessage,
        isUserMessage: data.isUserMessage,
        message: data.message,
        doctorId: data.doctorId,
        userId: data.userId,
      },
    });
  }

  async findChat(data: any) {
    const chat = await this.prismaSerice.chat.findMany({
      where: {
        doctorId: data.doctorId,
        userId: data.userId,
      },
      orderBy: {
        createdAt: "asc", // Ensure messages are sorted by time
      },
    });
    const formattedChat = chat.map((message) => ({
      time: message.createdAt,
      sender: message.isDoctorMessage ? "Doctor" : "User",
      message: message.message,
    }));

    console.log(formattedChat);
    return formattedChat;
  }

  async findDoctorChatList(doctorId: string) {
    const dcId = parseInt(doctorId, 10);
    const users = await this.prismaSerice.user.findMany({
      where: {
        Chat: {
          some: {
            doctorId: dcId, // Ensure the user has at least one chat with this doctor
          },
        },
      },
      select: {
        id: true,
        firstName: true,
        lastName: true,
        email: true,
      },
    });

    return users;
  }

  async findUserChatList(userId: string) {
    const uId = parseInt(userId, 10);

    const doctors = await this.prismaSerice.doctor.findMany({
      where: {
        Chat: {
          some: {
            userId: uId, // Ensure the doctor has at least one chat with this user
          },
        },
      },
      select: {
        id: true,
        name: true,
        email: true,
        position: true,
        address: true,
        contact: true,
      },
    });

    return doctors;
  }
}
